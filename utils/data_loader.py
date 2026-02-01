import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import json


class NERDataset(Dataset):
    def __init__(self, file_path: str, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.sentences, self.labels = self.load_data(file_path)
        self.label2id = {label: idx for idx, label in enumerate(config.LABELS)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # 构建词汇表
        self.build_vocab()

    def load_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences = []
        labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            label = []

            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        sentence.append(parts[0])
                        label.append(parts[1])
                else:
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []

            # 处理最后一个句子
            if sentence:
                sentences.append(sentence)
                labels.append(label)

        return sentences, labels

    def build_vocab(self):
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.vocab_size = len(self.word2id)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        # 转换为ID
        input_ids = [self.word2id.get(word, self.word2id['<UNK>']) for word in sentence]
        label_ids = [self.label2id[label] for label in labels]

        # 截断或填充
        if len(input_ids) > self.config.MAX_SEQ_LENGTH:
            input_ids = input_ids[:self.config.MAX_SEQ_LENGTH]
            label_ids = label_ids[:self.config.MAX_SEQ_LENGTH]
        else:
            padding_length = self.config.MAX_SEQ_LENGTH - len(input_ids)
            input_ids.extend([self.word2id['<PAD>']] * padding_length)
            label_ids.extend([self.label2id['O']] * padding_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.word2id['<PAD>'] else 0 for id in input_ids],
                                           dtype=torch.long),
            'original_sentence': sentence,
            'original_labels': labels
        }


def create_data_loader(file_path: str, config, batch_size: int, shuffle: bool = True):
    dataset = NERDataset(file_path, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'original_sentences': [item['original_sentence'] for item in batch],
        'original_labels': [item['original_labels'] for item in batch]
    }
