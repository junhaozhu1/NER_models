import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class SoftLexicon(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, dropout=0.3):
        super(SoftLexicon, self).__init__()

        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        # 软词典特征融合
        self.soft_attention = nn.Linear(embedding_dim * 2, 1)

        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 字符嵌入
        char_embeds = self.char_embedding(input_ids)

        # 词嵌入（简化：使用相同的输入）
        word_embeds = self.word_embedding(input_ids)

        # 软词典注意力
        combined_embeds = torch.cat([char_embeds, word_embeds], dim=-1)
        attention_scores = torch.sigmoid(self.soft_attention(combined_embeds))

        # 加权融合
        soft_lexicon_embeds = char_embeds + attention_scores * word_embeds

        # 拼接特征
        features = torch.cat([char_embeds, soft_lexicon_embeds], dim=-1)
        features = self.dropout(features)

        # LSTM编码
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out)

        logits = self.classifier(lstm_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class SoftLexiconModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = SoftLexicon(
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

        for batch in tqdm(train_loader, desc="Training SoftLexicon"):
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
            for batch in tqdm(test_loader, desc="Evaluating SoftLexicon"):
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
