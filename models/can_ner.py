import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_w = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, 1)

    def forward(self, char_hidden, word_hidden, mask=None):
        # char_hidden: (batch_size, char_len, hidden_dim)
        # word_hidden: (batch_size, word_len, hidden_dim)

        batch_size, char_len, hidden_dim = char_hidden.size()
        word_len = word_hidden.size(1)

        # 计算注意力分数
        char_transformed = self.W_c(char_hidden).unsqueeze(2)  # (batch, char_len, 1, hidden)
        word_transformed = self.W_w(word_hidden).unsqueeze(1)  # (batch, 1, word_len, hidden)

        scores = self.W_v(torch.tanh(char_transformed + word_transformed))  # (batch, char_len, word_len, 1)
        scores = scores.squeeze(-1)  # (batch, char_len, word_len)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # 计算注意力权重
        char_attention = torch.softmax(scores, dim=2)  # 对词维度softmax
        word_attention = torch.softmax(scores, dim=1)  # 对字符维度softmax

        # 加权求和
        char_context = torch.bmm(char_attention, word_hidden)  # (batch, char_len, hidden)
        word_context = torch.bmm(word_attention.transpose(1, 2), char_hidden)  # (batch, word_len, hidden)

        return char_context, word_context


class CANNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.3):
        super(CANNER, self).__init__()

        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)  # 简化：使用相同词表

        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 bidirectional=True, batch_first=True)
        self.word_lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 bidirectional=True, batch_first=True)

        self.co_attention = CoAttention(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 字符级编码
        char_embeds = self.char_embedding(input_ids)
        char_lstm_out, _ = self.char_lstm(char_embeds)

        # 词级编码（简化：使用相同输入）
        word_embeds = self.word_embedding(input_ids)
        word_lstm_out, _ = self.word_lstm(word_embeds)

        # Co-Attention
        char_context, word_context = self.co_attention(char_lstm_out, word_lstm_out)

        # 特征融合
        combined = torch.cat([char_lstm_out, char_context], dim=-1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class CANNERModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = CANNER(
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

        for batch in tqdm(train_loader, desc="Training CAN-NER"):
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
            for batch in tqdm(test_loader, desc="Evaluating CAN-NER"):
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
