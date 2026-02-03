import torch
import torch.nn as nn
from transformers import BertModel
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class LEBERT(nn.Module):
    def __init__(self, bert_model_name, vocab_size, embedding_dim, num_labels, dropout=0.1):
        super(LEBERT, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lexicon_embedding = nn.Embedding(vocab_size, embedding_dim)

        # 词典特征适配层
        self.lexicon_adapter = nn.Linear(embedding_dim, self.bert.config.hidden_size)

        # 特征融合
        self.fusion_layer = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, lexicon_ids=None, labels=None):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden = bert_outputs.last_hidden_state

        # 词典特征
        if lexicon_ids is None:
            lexicon_ids = input_ids  # 简化：使用相同的输入

        lexicon_embeds = self.lexicon_embedding(lexicon_ids)
        lexicon_features = self.lexicon_adapter(lexicon_embeds)

        # 特征融合
        combined_features = torch.cat([bert_hidden, lexicon_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = torch.relu(fused_features)
        fused_features = self.dropout(fused_features)

        logits = self.classifier(fused_features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class LEBERTModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = LEBERT(
            bert_model_name=self.config.BERT_MODEL_NAME,
            vocab_size=vocab_size,
            embedding_dim=self.config.EMBEDDING_DIM,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training LEBERT"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss = self.model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating LEBERT"):
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
