import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class NERDataset(Dataset):
    def __init__(self, file_path, word2idx=None, label2idx=None, max_len=128):
        self.sentences = []
        self.labels = []
        self.max_len = max_len
        
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if line:
                    word, tag = line.split()
                    words.append(word)
                    tags.append(tag)
                elif words:  # 空行表示句子结束
                    self.sentences.append(words[:max_len])
                    self.labels.append(tags[:max_len])
                    words, tags = [], []
        
        # 构建词表
        if word2idx is None:
            self.word2idx = self._build_vocab(self.sentences)
            self.label2idx = self._build_label_vocab(self.labels)
        else:
            self.word2idx = word2idx
            self.label2idx = label2idx
        
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
    def _build_vocab(self, sentences, min_freq=2):
        word_freq = Counter()
        for sent in sentences:
            word_freq.update(sent)
        
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                word2idx[word] = len(word2idx)
        
        return word2idx
    
    def _build_label_vocab(self, labels_list):
        label_set = set()
        for labels in labels_list:
            label_set.update(labels)
        
        label2idx = {label: idx for idx, label in enumerate(sorted(label_set))}
        return label2idx
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]
        
        # 转换为索引
        word_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        label_ids = [self.label2idx[l] for l in labels]
        
        return word_ids, label_ids, len(word_ids)

def pad_batch(batch):
    """批处理函数"""
    word_ids_list, label_ids_list, lengths = zip(*batch)
    max_len = max(lengths)
    
    # Padding
    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    label_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, (word_id, label_id, length) in enumerate(zip(word_ids_list, label_ids_list, lengths)):
        word_ids[i, :length] = torch.LongTensor(word_id)
        label_ids[i, :length] = torch.LongTensor(label_id)
        mask[i, :length] = True
    
    return word_ids, label_ids, mask, torch.LongTensor(lengths)

def get_dataloader(file_path, batch_size=32, shuffle=True, word2idx=None, label2idx=None):
    dataset = NERDataset(file_path, word2idx, label2idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_batch), dataset
