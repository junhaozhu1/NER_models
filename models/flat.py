import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output


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
        self.pos_embedding = PositionEmbedding(embedding_dim)

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 嵌入层
        seq_len = input_ids.size(1)
        embeddings = self.embedding(input_ids)
        pos_embeddings = self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        embeddings = embeddings + pos_embeddings.unsqueeze(0)
        embeddings = self.dropout(embeddings)

        # Transformer编码
        hidden_states = embeddings.transpose(0, 1)  # (seq_len, batch_size, d_model)

        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask.bool())

        hidden_states = hidden_states.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # 分类
        logits = self.classifier(hidden_states)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class FLATModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = FLAT(
            vocab_size=vocab_size,
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_labels=num_labels,
            n_heads=8,
            n_layers=4,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
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
            loss.backward()
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
                    for j in range(len(original_labels[i])):
                        if j < pred_seq.size(0):
                            pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j].item()])
                        else:
                            pred_label_seq.append('O')
                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
