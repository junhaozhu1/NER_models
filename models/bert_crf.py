import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
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
            # 确保labels在有效范围内
            labels = labels.masked_fill(labels < 0, 0)
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions


class BERTCRFModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL_NAME)

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
        original_labels = batch['original_labels']

        # 首先，将句子列表转换为字符列表（因为中文BERT是基于字符的）
        char_sentences = []
        char_labels = []

        for sent, labels in zip(sentences, original_labels):
            chars = []
            char_label_list = []

            for word, label in zip(sent, labels):
                # 将每个词拆分成字符
                word_chars = list(word)
                chars.extend(word_chars)

                # 为每个字符分配标签
                if label.startswith('B-'):
                    # B-标签只给第一个字符
                    char_label_list.append(label)
                    # 其余字符用I-标签
                    char_label_list.extend(['I-' + label[2:]] * (len(word_chars) - 1))
                elif label.startswith('I-'):
                    # I-标签给所有字符
                    char_label_list.extend([label] * len(word_chars))
                else:  # O标签
                    char_label_list.extend(['O'] * len(word_chars))

            char_sentences.append(chars)
            char_labels.append(char_label_list)

        # 使用BERT tokenizer编码
        encoded = self.tokenizer(
            char_sentences,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQ_LENGTH,
            return_tensors='pt',
            is_split_into_words=True
        )

        # 创建label2id映射
        label2id = {label: idx for idx, label in enumerate(self.config.LABELS)}

        # 对齐标签
        aligned_labels = []
        for i in range(len(char_sentences)):
            word_ids = encoded.word_ids(i)
            aligned_label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_label_ids.append(-100)  # 忽略特殊token
                else:
                    if word_idx < len(char_labels[i]):
                        label = char_labels[i][word_idx]
                        aligned_label_ids.append(label2id.get(label, label2id['O']))
                    else:
                        aligned_label_ids.append(label2id['O'])

            aligned_labels.append(aligned_label_ids)

        # 将-100替换为0（CRF不支持负数索引）
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        labels_tensor[labels_tensor == -100] = label2id['O']

        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'labels': labels_tensor.to(self.device),
            'word_ids': [encoded.word_ids(i) for i in range(len(sentences))],
            'original_labels': original_labels,
            'char_labels': char_labels,
            'char_sentences': char_sentences
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        id2label = {idx: label for idx, label in enumerate(self.config.LABELS)}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating BERT-CRF"):
                prepared_batch = self.prepare_batch(batch)
                original_labels = prepared_batch['original_labels']
                original_sentences = batch['original_sentences']

                start_time = time.time()
                predictions = self.model(
                    prepared_batch['input_ids'],
                    prepared_batch['attention_mask']
                )
                batch_time = time.time() - start_time

                # 将字符级预测转换回词级
                pred_labels = []
                for i, (pred_seq, sent) in enumerate(zip(predictions, original_sentences)):
                    word_labels = []
                    char_idx = 0

                    # 获取有效的预测（排除特殊token）
                    word_ids = prepared_batch['word_ids'][i]
                    valid_preds = []
                    for j, (word_id, pred_id) in enumerate(zip(word_ids, pred_seq)):
                        if word_id is not None:
                            valid_preds.append(id2label[pred_id])

                    # 将字符级标签转换为词级标签
                    for word in sent:
                        if char_idx < len(valid_preds):
                            # 取词的第一个字符的标签作为整个词的标签
                            word_label = valid_preds[char_idx]
                            word_labels.append(word_label)
                            char_idx += len(word)
                        else:
                            word_labels.append('O')

                    # 确保长度匹配
                    while len(word_labels) < len(original_labels[i]):
                        word_labels.append('O')
                    word_labels = word_labels[:len(original_labels[i])]

                    pred_labels.append(word_labels)

                metrics.add_batch(pred_labels, original_labels, batch_time)

        return metrics.compute()
