import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm
import math


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0)]


class FLAT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels,
                 n_heads=8, n_layers=4, dropout=0.1):
        super(FLAT, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 调整维度以确保能被n_heads整除
        # 找到最接近hidden_dim且能被n_heads整除的值
        self.d_model = (hidden_dim // n_heads) * n_heads

        # 如果embedding_dim与d_model不同，需要投影层
        if embedding_dim != self.d_model:
            self.input_projection = nn.Linear(embedding_dim, self.d_model)
        else:
            self.input_projection = None

        self.pos_embedding = PositionEmbedding(self.d_model)

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,  # 通常是d_model的4倍
                dropout=dropout,
                activation='relu',
                batch_first=True  # 使用batch_first格式
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 嵌入层
        batch_size, seq_len = input_ids.size()
        embeddings = self.embedding(input_ids)

        # 投影到正确的维度
        if self.input_projection is not None:
            embeddings = self.input_projection(embeddings)

        # 添加位置编码
        pos_embeddings = self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        embeddings = embeddings + pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = self.dropout(embeddings)

        # 准备attention mask (True表示要mask的位置)
        # TransformerEncoderLayer期望的mask: True表示忽略的位置
        src_key_padding_mask = ~attention_mask.bool()

        # Transformer编码
        hidden_states = embeddings

        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=src_key_padding_mask
            )

        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)

        # 分类
        logits = self.classifier(hidden_states)

        if labels is not None:
            # 只计算有效位置的loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = nn.CrossEntropyLoss()(active_logits, active_labels)
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class FLATModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        # 计算合适的注意力头数
        # 确保hidden_dim能被n_heads整除
        n_heads = 8
        while self.config.HIDDEN_DIM % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.model = FLAT(
            vocab_size=vocab_size,
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_labels=num_labels,
            n_heads=n_heads,
            n_layers=4,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)

        print(f"FLAT model initialized with n_heads={n_heads}, d_model={self.model.d_model}")

        return self.model

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training FLAT"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss = self.model(input_ids, attention_mask, labels)

            # 梯度裁剪防止梯度爆炸
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating FLAT"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                original_labels = batch['original_labels']

                start_time = time.time()
                predictions = self.model(input_ids, attention_mask)
                batch_time = time.time() - start_time

                # 转换预测结果
                pred_labels = []
                for i, pred_seq in enumerate(predictions):
                    pred_label_seq = []
                    original_length = len(original_labels[i])

                    for j in range(original_length):
                        if j < pred_seq.size(0) and attention_mask[i][j] == 1:
                            pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j].item()])
                        else:
                            pred_label_seq.append('O')

                    # 确保长度匹配
                    pred_label_seq = pred_label_seq[:original_length]
                    while len(pred_label_seq) < original_length:
                        pred_label_seq.append('O')

                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
