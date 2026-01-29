import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_heads, head_dim)
        query = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.dense(context)
        return output


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(channels * len(kernel_sizes))

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)

        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, channels * len(kernel_sizes), seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels * len(kernel_sizes))
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x


class W2NER(nn.Module):
    def __init__(self,
                 bert_model_name='bert-base-chinese',
                 num_labels=7,
                 dist_emb_dim=64,
                 hidden_size=768,
                 conv_channels=128,
                 num_heads=8,
                 dropout=0.3):
        super().__init__()

        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name,
                                              add_pooling_layer=False,
                                              return_dict=True)

        # 位置距离嵌入
        self.dist_embeddings = nn.Embedding(512, dist_emb_dim)

        # 多头注意力层
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)

        # 卷积层
        self.conv_layer = ConvolutionLayer(hidden_size, conv_channels, dropout=dropout)
        conv_output_size = conv_channels * 3  # 3个卷积核

        # 词对表示的融合层
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden_size * 2 + conv_output_size + dist_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 关系分类层
        self.classifier = nn.Linear(hidden_size // 2, num_labels * num_labels)

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

    def get_distance_embeddings(self, seq_len, device):
        """生成位置距离嵌入"""
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance_mat = distance_mat.clamp(-255, 255) + 255
        return self.dist_embeddings(distance_mat)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape

        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 自注意力增强
        attention_output = self.self_attention(sequence_output, attention_mask)
        sequence_output = sequence_output + attention_output

        # 卷积特征
        conv_output = self.conv_layer(sequence_output)  # (batch_size, seq_len, conv_output_size)

        # 获取距离嵌入
        dist_emb = self.get_distance_embeddings(seq_len, input_ids.device)  # (seq_len, seq_len, dist_emb_dim)
        dist_emb = dist_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 构建词对表示
        # 扩展sequence_output和conv_output用于词对
        seq_i = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (batch, seq_len, seq_len, hidden)
        seq_j = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (batch, seq_len, seq_len, hidden)
        conv_i = conv_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        conv_j = conv_output.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # 拼接所有特征
        pair_features = torch.cat([
            seq_i,
            seq_j,
            conv_i + conv_j,  # 卷积特征的组合
            dist_emb
        ], dim=-1)

        # 词对表示投影
        pair_hidden = self.pair_proj(pair_features)  # (batch, seq_len, seq_len, hidden//2)

        # 关系分类
        relation_scores = self.classifier(pair_hidden)  # (batch, seq_len, seq_len, num_labels^2)
        relation_scores = relation_scores.view(batch_size, seq_len, seq_len, self.num_labels, self.num_labels)

        return relation_scores

    def loss(self, input_ids, label_ids, attention_mask, lengths=None):
        """
        计算损失函数（统一接口，忽略lengths参数）

        Args:
            input_ids: 输入token ids
            label_ids: 标签ids
            attention_mask: 注意力掩码
            lengths: 序列长度（W2NER不使用，但为了统一接口保留）
        """
        batch_size, seq_len = input_ids.shape

        # 获取预测分数
        relation_scores = self.forward(input_ids, attention_mask)

        # 将序列标签转换为词对关系矩阵
        relation_labels = self.build_relation_matrix(label_ids, attention_mask)

        # 计算损失
        relation_scores = relation_scores.view(-1, self.num_labels * self.num_labels)
        relation_labels = relation_labels.view(-1)

        # 只计算有效位置的损失
        valid_mask = (relation_labels != -100)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device)

        loss = F.cross_entropy(
            relation_scores[valid_mask],
            relation_labels[valid_mask]
        )

        return loss

    def build_relation_matrix(self, label_ids, attention_mask):
        """
        将序列标签转换为词对关系矩阵
        """
        batch_size, seq_len = label_ids.shape
        relation_matrix = torch.full(
            (batch_size, seq_len, seq_len),
            fill_value=-100,
            dtype=torch.long,
            device=label_ids.device
        )

        for b in range(batch_size):
            valid_len = attention_mask[b].sum().item()

            # 对于每个有效位置
            for i in range(valid_len):
                for j in range(valid_len):
                    label_i = label_ids[b, i].item()
                    label_j = label_ids[b, j].item()

                    # 计算关系标签
                    # 这里简化处理：如果i和j属于同一个实体，则关系为label_i * num_labels + label_j
                    # 否则为0（表示无关系）
                    if self.is_same_entity(label_ids[b], i, j, valid_len):
                        relation_matrix[b, i, j] = label_i * self.num_labels + label_j
                    else:
                        relation_matrix[b, i, j] = 0  # O标签

        return relation_matrix

    def is_same_entity(self, labels, i, j, valid_len):
        """判断位置i和j是否属于同一个实体"""
        if i >= valid_len or j >= valid_len:
            return False

        # 简单规则：如果i和j相邻且都不是O标签，则属于同一实体
        if abs(i - j) <= 1 and labels[i] != 0 and labels[j] != 0:
            return True

        # 更复杂的规则可以根据BIO标签体系来判断
        return False

    def predict(self, input_ids, attention_mask, lengths=None):
        """
        预测函数（统一接口，忽略lengths参数）

        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            lengths: 序列长度（W2NER不使用，但为了统一接口保留）
        """
        self.eval()
        with torch.no_grad():
            relation_scores = self.forward(input_ids, attention_mask)

            # 从关系矩阵中提取序列标签
            batch_size, seq_len = input_ids.shape
            predictions = []

            for b in range(batch_size):
                valid_len = attention_mask[b].sum().item()
                pred_labels = self.decode_relation_matrix(
                    relation_scores[b, :valid_len, :valid_len]
                )
                predictions.append(pred_labels)

            return predictions

    def decode_relation_matrix(self, relation_matrix):
        """从关系矩阵中解码出序列标签"""
        seq_len = relation_matrix.size(0)
        relation_preds = relation_matrix.argmax(dim=-1).argmax(dim=-1)  # (seq_len, seq_len)

        # 简单解码策略：使用对角线上的预测
        sequence_labels = []
        for i in range(seq_len):
            # 获取第i个词的标签（通过投票或其他策略）
            label_votes = relation_preds[i, :] // self.num_labels
            # 简单地使用最常见的标签
            label = label_votes[i].item() % self.num_labels
            sequence_labels.append(label)

        return sequence_labels
