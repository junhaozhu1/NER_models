import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import math
import numpy as np


class SinusoidalPositionEncoder(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionWithRelativePosition(nn.Module):
    """带相对位置编码的多头注意力"""

    def __init__(self, d_model, n_heads, dropout=0.1, max_relative_position=16):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position

        # 查询、键、值的投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        # 相对位置嵌入
        self.relative_positions_encoding = nn.Embedding(
            2 * max_relative_position + 1, self.d_k
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None, relative_positions=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 残差连接
        residual = query

        # 投影并reshape为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加相对位置信息
        if relative_positions is not None:
            # 将相对位置转换为嵌入索引
            relative_positions_clipped = torch.clamp(
                relative_positions,
                -self.max_relative_position,
                self.max_relative_position
            )
            relative_positions_clipped = relative_positions_clipped + self.max_relative_position

            # 获取相对位置嵌入
            rel_embeddings = self.relative_positions_encoding(relative_positions_clipped)

            # 计算相对位置注意力
            # [batch, heads, seq_len, seq_len, d_k]
            rel_scores = torch.einsum('bhid,bijde->bhije', Q, rel_embeddings)
            scores = scores + rel_scores.mean(dim=-1)

        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力到值
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 最终投影
        output = self.fc(context)
        output = self.dropout(output)

        # 添加残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attention


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRelativePosition(
            d_model, n_heads, dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, relative_positions=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, mask, relative_positions)

        # 前馈网络
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(attn_output + ff_output)

        return output


class FLAT(nn.Module):
    """FLAT模型实现"""

    def __init__(
            self,
            vocab_size,
            num_labels,
            d_model=256,
            n_heads=8,
            n_layers=2,
            d_ff=1024,
            dropout=0.3,
            max_len=512,
            use_bigram=True,
            bigram_vocab_size=None
    ):
        super().__init__()

        # 字符嵌入
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 双字符嵌入（可选）
        self.use_bigram = use_bigram
        if use_bigram and bigram_vocab_size:
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, d_model, padding_idx=0)
            self.combine_embeddings = nn.Linear(d_model * 2, d_model)

        # 位置编码
        self.position_encoder = SinusoidalPositionEncoder(d_model, max_len)

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def compute_relative_positions(self, seq_len, device):
        """计算相对位置矩阵"""
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        return distance_mat.unsqueeze(0)  # [1, seq_len, seq_len]

    def forward(self, char_ids, bigram_ids=None, mask=None):
        """
        前向传播

        Args:
            char_ids: 字符ID序列 [batch_size, seq_len]
            bigram_ids: 双字符ID序列 [batch_size, seq_len] (可选)
            mask: 掩码 [batch_size, seq_len]
        """
        batch_size, seq_len = char_ids.size()

        # 字符嵌入
        char_embeds = self.char_embedding(char_ids)

        # 如果使用双字符嵌入
        if self.use_bigram and bigram_ids is not None:
            bigram_embeds = self.bigram_embedding(bigram_ids)
            # 结合字符和双字符嵌入
            combined_embeds = torch.cat([char_embeds, bigram_embeds], dim=-1)
            embeddings = self.combine_embeddings(combined_embeds)
        else:
            embeddings = char_embeds

        # 添加位置编码
        embeddings = self.position_encoder(embeddings)
        embeddings = self.dropout(embeddings)

        # 计算相对位置
        relative_positions = self.compute_relative_positions(seq_len, char_ids.device)
        relative_positions = relative_positions.expand(batch_size, seq_len, seq_len)

        # Transformer编码
        output = embeddings
        for layer in self.transformer_layers:
            output = layer(output, mask, relative_positions)

        # 分类
        output = self.dropout(output)
        emissions = self.classifier(output)

        return emissions

    def loss(self, char_ids, label_ids, mask, bigram_ids=None):
        """计算损失"""
        emissions = self.forward(char_ids, bigram_ids, mask)
        mask = mask.bool()
        return -self.crf(emissions, label_ids, mask=mask, reduction='mean')

    def predict(self, char_ids, mask, bigram_ids=None):
        """预测标签序列"""
        emissions = self.forward(char_ids, bigram_ids, mask)
        mask = mask.bool()
        return self.crf.decode(emissions, mask=mask)


class FLATWithLexicon(FLAT):
    """带词典的FLAT模型"""

    def __init__(
            self,
            vocab_size,
            num_labels,
            lexicon_size,
            d_model=256,
            n_heads=8,
            n_layers=2,
            d_ff=1024,
            dropout=0.3,
            max_len=512
    ):
        super().__init__(
            vocab_size, num_labels, d_model, n_heads,
            n_layers, d_ff, dropout, max_len, use_bigram=False
        )

        # 词汇嵌入
        self.word_embedding = nn.Embedding(lexicon_size, d_model, padding_idx=0)

        # 词汇位置嵌入（开始位置和结束位置）
        self.word_pos_embedding = nn.Embedding(max_len * 2, d_model)

        # 融合层
        self.fusion_layer = nn.Linear(d_model * 2, d_model)

    def forward_with_lexicon(
            self,
            char_ids,
            word_ids,
            word_positions,
            mask=None
    ):
        """
        带词典信息的前向传播

        Args:
            char_ids: 字符ID [batch_size, seq_len]
            word_ids: 匹配到的词ID [batch_size, seq_len, max_words]
            word_positions: 词的位置信息 [batch_size, seq_len, max_words, 2]
            mask: 字符级掩码
        """
        batch_size, seq_len = char_ids.size()

        # 获取字符嵌入
        char_embeds = self.char_embedding(char_ids)
        char_embeds = self.position_encoder(char_embeds)

        # 获取词汇嵌入
        word_embeds = self.word_embedding(word_ids)  # [batch, seq_len, max_words, d_model]

        # 计算词汇的位置嵌入
        start_pos = word_positions[..., 0]  # 开始位置
        end_pos = word_positions[..., 1]  # 结束位置

        start_embeds = self.word_pos_embedding(start_pos)
        end_embeds = self.word_pos_embedding(end_pos)

        # 融合位置信息
        word_embeds = word_embeds + start_embeds + end_embeds

        # 聚合词汇信息（平均池化）
        word_mask = (word_ids != 0).float()
        word_embeds = (word_embeds * word_mask.unsqueeze(-1)).sum(dim=2)
        word_embeds = word_embeds / (word_mask.sum(dim=2, keepdim=True) + 1e-9)

        # 融合字符和词汇嵌入
        combined = torch.cat([char_embeds, word_embeds], dim=-1)
        embeddings = self.fusion_layer(combined)
        embeddings = self.dropout(embeddings)

        # Transformer编码
        relative_positions = self.compute_relative_positions(seq_len, char_ids.device)
        relative_positions = relative_positions.expand(batch_size, seq_len, seq_len)

        output = embeddings
        for layer in self.transformer_layers:
            output = layer(output, mask, relative_positions)

        # 分类
        output = self.dropout(output)
        emissions = self.classifier(output)

        return emissions
