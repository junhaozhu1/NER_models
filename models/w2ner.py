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

        # 多尺度CNN
        self.convs = nn.ModuleList([
            ConvolutionLayer(embedding_dim, hidden_dim // len(kernel_sizes), k, dropout)
            for k in kernel_sizes
        ])

        # BiLSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2,
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
        attn_out, _ = self.word_word_attention(lstm_out, lstm_out, lstm_out,
                                               key_padding_mask=~attention_mask.bool())

        # 融合特征
        combined = torch.cat([lstm_out, attn_out], dim=-1)
        logits = self.classifier(combined)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
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
                    for j in range(len(original_labels[i])):
                        if j < pred_seq.size(0):
                            pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j].item()])
                        else:
                            pred_label_seq.append('O')
                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
