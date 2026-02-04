import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, kernel_size, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Conv1d(input_size, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.conv(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
        return x


class W2NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels,
                 kernel_sizes=[3, 5, 7], dropout=0.3):
        super(W2NER, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # 计算每个CNN的输出维度，确保总和等于hidden_dim
        n_kernels = len(kernel_sizes)
        # 基础维度
        base_dim = hidden_dim // n_kernels
        # 剩余维度分配给第一个kernel
        remainder = hidden_dim % n_kernels

        conv_dims = [base_dim] * n_kernels
        conv_dims[0] += remainder  # 将剩余维度加到第一个

        # 多尺度CNN
        self.convs = nn.ModuleList([
            ConvolutionLayer(embedding_dim, conv_dims[i], k, dropout)
            for i, k in enumerate(kernel_sizes)
        ])

        # 验证总维度
        self.total_conv_dim = sum(conv_dims)
        assert self.total_conv_dim == hidden_dim, f"Conv dimensions sum {self.total_conv_dim} != hidden_dim {hidden_dim}"

        # BiLSTM - 输入维度现在正确了
        self.lstm = nn.LSTM(self.total_conv_dim, hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        # Word-Word关系建模
        self.word_word_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # 分类器
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 嵌入
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # 多尺度CNN
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(embeddings))
        conv_out = torch.cat(conv_outputs, dim=-1)

        # BiLSTM
        lstm_out, _ = self.lstm(conv_out)

        # Word-Word关系
        attn_out, _ = self.word_word_attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=~attention_mask.bool()
        )

        # 融合特征
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        logits = self.classifier(combined)

        if labels is not None:
            # 创建mask来忽略padding位置的loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = nn.CrossEntropyLoss()(active_logits, active_labels)
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class W2NERModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = W2NER(
            vocab_size=vocab_size,
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training W2NER"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss = self.model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating W2NER"):
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

                    # 只取实际长度的预测
                    for j in range(original_length):
                        if j < pred_seq.size(0) and attention_mask[i][j] == 1:
                            pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j].item()])
                        else:
                            pred_label_seq.append('O')

                    # 确保长度匹配
                    assert len(pred_label_seq) == len(original_labels[i]), \
                        f"Length mismatch: pred={len(pred_label_seq)}, true={len(original_labels[i])}"

                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
