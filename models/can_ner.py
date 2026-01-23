import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import math


class NonLocalAttention(nn.Module):
    """非局部注意力模块"""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value 投影层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 计算 Q, K, V
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用mask
        if mask is not None:
            # 扩展mask维度以匹配注意力分数
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 输出投影
        output = self.output_proj(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        return self.layer_norm(x + output)


class ContextAwareLayer(nn.Module):
    """上下文感知层"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )

    def forward(self, local_features, global_features):
        # 拼接局部和全局特征
        combined = torch.cat([local_features, global_features], dim=-1)

        # 计算门控值
        gate_values = self.gate(combined)

        # 计算融合特征
        fused_features = self.context_fusion(combined)

        # 应用门控机制
        output = gate_values * local_features + (1 - gate_values) * fused_features

        return output


class CANNER(nn.Module):
    """Context-Aware Non-local Attention for NER"""

    def __init__(self, vocab_size, num_labels, embedding_dim=100, hidden_dim=256,
                 num_layers=2, num_heads=8, dropout=0.5):
        super().__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)

        # BiLSTM编码器
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 非局部注意力层
        self.non_local_attention = NonLocalAttention(hidden_dim, num_heads)

        # 上下文感知层
        self.context_aware = ContextAwareLayer(hidden_dim)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 分类层
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # CRF层
        self.crf = CRF(num_labels, batch_first=True)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode(self, word_ids, mask, lengths=None):
        """编码输入序列"""
        # 词嵌入
        embeddings = self.embedding(word_ids)
        embeddings = self.embedding_dropout(embeddings)

        # BiLSTM编码
        if lengths is not None:
            # 使用PackedSequence优化
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed_embeddings = pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_embeddings)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(embeddings)

        return lstm_out

    def forward(self, word_ids, mask=None, lengths=None):
        """前向传播"""
        # 编码输入
        local_features = self.encode(word_ids, mask, lengths)

        # 应用非局部注意力
        global_features = self.non_local_attention(local_features, mask)

        # 上下文感知融合
        context_aware_features = self.context_aware(local_features, global_features)

        # 特征融合
        fused_features = self.feature_fusion(context_aware_features)

        # 分类
        emissions = self.classifier(fused_features)

        return emissions

    def loss(self, word_ids, label_ids, mask, lengths=None):
        """计算损失"""
        emissions = self.forward(word_ids, mask, lengths)
        return -self.crf(emissions, label_ids, mask=mask, reduction='mean')

    def predict(self, word_ids, mask, lengths=None):
        """预测标签"""
        emissions = self.forward(word_ids, mask, lengths)
        return self.crf.decode(emissions, mask=mask)

    def get_attention_weights(self, word_ids, mask, lengths=None):
        """获取注意力权重（用于可视化）"""
        with torch.no_grad():
            local_features = self.encode(word_ids, mask, lengths)
            batch_size, seq_len, _ = local_features.size()

            # 获取注意力层的Q, K
            Q = self.non_local_attention.query_proj(local_features)
            K = self.non_local_attention.key_proj(local_features)

            Q = Q.view(batch_size, seq_len, self.non_local_attention.num_heads,
                       self.non_local_attention.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.non_local_attention.num_heads,
                       self.non_local_attention.head_dim).transpose(1, 2)

            # 计算注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.non_local_attention.head_dim)

            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask_expanded == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)

            # 平均所有头的注意力权重
            attn_weights = attn_weights.mean(dim=1)

            return attn_weights


class CANNERWithBERT(nn.Module):
    """使用BERT作为编码器的CAN-NER模型"""

    def __init__(self, bert_model_name='bert-base-chinese', num_labels=7,
                 num_heads=8, dropout=0.3):
        super().__init__()

        from transformers import BertModel

        # BERT编码器
        self.bert = BertModel.from_pretrained(
            bert_model_name,
            add_pooling_layer=False,
            return_dict=True
        )

        hidden_dim = self.bert.config.hidden_size

        # 非局部注意力层
        self.non_local_attention = NonLocalAttention(hidden_dim, num_heads)

        # 上下文感知层
        self.context_aware = ContextAwareLayer(hidden_dim)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 分类层
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # CRF层
        self.crf = CRF(num_labels, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        local_features = outputs.last_hidden_state
        local_features = self.dropout(local_features)

        # 应用非局部注意力
        global_features = self.non_local_attention(local_features, attention_mask)

        # 上下文感知融合
        context_aware_features = self.context_aware(local_features, global_features)

        # 特征融合
        fused_features = self.feature_fusion(context_aware_features)

        # 分类
        emissions = self.classifier(fused_features)

        return emissions

    def loss(self, input_ids, label_ids, attention_mask):
        """计算损失"""
        emissions = self.forward(input_ids, attention_mask)
        mask = attention_mask.bool()
        return -self.crf(emissions, label_ids, mask=mask, reduction='mean')

    def predict(self, input_ids, attention_mask):
        """预测标签"""
        emissions = self.forward(input_ids, attention_mask)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)
