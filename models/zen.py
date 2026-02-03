import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class ZEN(nn.Module):
    def __init__(self, bert_model_name, num_labels, n_gram_size=2, dropout=0.1):
        super(ZEN, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # N-gram编码器
        self.ngram_embedding = nn.Embedding(10000, hidden_size)  # 简化的n-gram词表
        self.ngram_encoder = nn.LSTM(hidden_size, hidden_size // 2,
                                     bidirectional=True, batch_first=True)

        # 融合层
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def extract_ngrams(self, input_ids, n=2):
        """提取n-gram特征（简化版本）"""
        batch_size, seq_len = input_ids.size()
        ngram_ids = torch.zeros_like(input_ids)

        for i in range(seq_len - n + 1):
            # 简化：使用位置作为n-gram的ID
            ngram_ids[:, i] = i % 10000

        return ngram_ids

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden = bert_outputs.last_hidden_state

        # N-gram编码
        ngram_ids = self.extract_ngrams(input_ids)
        ngram_embeds = self.ngram_embedding(ngram_ids)
        ngram_hidden, _ = self.ngram_encoder(ngram_embeds)

        # 特征融合
        combined = torch.cat([bert_hidden, ngram_hidden], dim=-1)
        fused = self.fusion_activation(self.fusion_layer(combined))
        fused = self.dropout(fused)

        logits = self.classifier(fused)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class ZENModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = ZEN(
            bert_model_name=self.config.BERT_MODEL_NAME,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training ZEN"):
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
            for batch in tqdm(test_loader, desc="Evaluating ZEN"):
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
