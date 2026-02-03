import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class BERTCRF(nn.Module):
    def __init__(self, bert_model_name, num_labels, dropout=0.1):
        super(BERTCRF, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions


class BERTCRFModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = BERTCRF(
            bert_model_name=self.config.BERT_MODEL_NAME,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def prepare_batch(self, batch):
        """为BERT准备批次数据"""
        sentences = batch['original_sentences']

        # 使用BERT tokenizer编码
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_tensors='pt',
            is_split_into_words=True  # 因为输入已经是分词的
        )

        # 对齐标签
        aligned_labels = []
        for i, labels in enumerate(batch['original_labels']):
            word_ids = encoded.word_ids(i)
            aligned_label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:  # 特殊token
                    aligned_label_ids.append(0)  # O标签
                elif word_idx != previous_word_idx:  # 新词的第一个子词
                    label = labels[word_idx] if word_idx < len(labels) else 'O'
                    aligned_label_ids.append(batch['labels'][0].new_tensor(
                        self.model.crf.num_tags - 1 if label == 'O' else
                        [i for i, l in enumerate(self.config.LABELS) if l == label][0]
                    ).item())
                else:  # 同一个词的其他子词
                    aligned_label_ids.append(aligned_label_ids[-1])
                previous_word_idx = word_idx

            aligned_labels.append(aligned_label_ids)

        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'labels': torch.tensor(aligned_labels, dtype=torch.long).to(self.device),
            'word_ids': [encoded.word_ids(i) for i in range(len(sentences))]
        }

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training BERT-CRF"):
            optimizer.zero_grad()

            prepared_batch = self.prepare_batch(batch)
            loss = self.model(
                prepared_batch['input_ids'],
                prepared_batch['attention_mask'],
                prepared_batch['labels']
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating BERT-CRF"):
                prepared_batch = self.prepare_batch(batch)
                original_labels = batch['original_labels']

                start_time = time.time()
                predictions = self.model(
                    prepared_batch['input_ids'],
                    prepared_batch['attention_mask']
                )
                batch_time = time.time() - start_time

                # 将预测结果映射回原始词
                pred_labels = []
                for i, (pred_seq, word_ids) in enumerate(zip(predictions, prepared_batch['word_ids'])):
                    pred_label_seq = []
                    previous_word_idx = None

                    for j, word_idx in enumerate(word_ids):
                        if word_idx is not None and word_idx != previous_word_idx:
                            if j < len(pred_seq):
                                pred_label_seq.append(test_loader.dataset.id2label[pred_seq[j]])
                        previous_word_idx = word_idx

                    # 确保长度匹配
                    while len(pred_label_seq) < len(original_labels[i]):
                        pred_label_seq.append('O')
                    pred_label_seq = pred_label_seq[:len(original_labels[i])]

                    pred_labels.append(pred_label_seq)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
