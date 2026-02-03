import torch
import torch.nn as nn
from torchcrf import CRF
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.5):
        super(BiLSTMCRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=2, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Embedding层
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # LSTM层
        lstm_out, _ = self.lstm(embeddings)

        # 线性层
        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            # 训练时计算loss
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            # 推理时返回预测结果
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions


class BiLSTMCRFModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = BiLSTMCRF(
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

        for batch in tqdm(train_loader, desc="Training"):
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
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                original_labels = batch['original_labels']

                start_time = time.time()
                predictions = self.model(input_ids, attention_mask)
                batch_time = time.time() - start_time

                # 转换预测结果，确保与原始标签长度一致
                pred_labels = []
                for i, pred_seq in enumerate(predictions):
                    # 获取原始句子的实际长度
                    original_length = len(original_labels[i])

                    # 确保预测标签与原始标签长度完全一致
                    pred_label_seq = []
                    for j in range(original_length):
                        if j < len(pred_seq):
                            pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j]])
                        else:
                            # 如果预测序列比原始序列短，用'O'填充
                            pred_label_seq.append('O')

                    # 确保长度完全匹配
                    assert len(pred_label_seq) == len(original_labels[i]), \
                        f"Length mismatch: pred={len(pred_label_seq)}, true={len(original_labels[i])}"

                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
