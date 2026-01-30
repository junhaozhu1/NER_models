import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FLATDataset(Dataset):
    """FLAT模型的数据集类"""

    def __init__(
            self,
            file_path,
            word2idx=None,
            label2idx=None,
            bigram2idx=None,
            max_len=128,
            create_bigram=True
    ):
        self.max_len = max_len
        self.create_bigram = create_bigram

        # 构建词表
        if word2idx is None:
            self.word2idx = {'<PAD>': 0, '<UNK>': 1}
            self.label2idx = {'O': 0}
            if create_bigram:
                self.bigram2idx = {'<PAD>': 0, '<UNK>': 1}
        else:
            self.word2idx = word2idx
            self.label2idx = label2idx
            self.bigram2idx = bigram2idx if bigram2idx else {'<PAD>': 0, '<UNK>': 1}

        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # 读取数据
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        """加载数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        data.append((words, labels))
                        words, labels = [], []
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        word, label = parts
                        words.append(word)
                        labels.append(label)

                        # 更新词表
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)
                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

        # 构建双字符词表
        if self.create_bigram and hasattr(self, 'bigram2idx'):
            for words, _ in data:
                for i in range(len(words) - 1):
                    bigram = words[i] + words[i + 1]
                    if bigram not in self.bigram2idx:
                        self.bigram2idx[bigram] = len(self.bigram2idx)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, labels = self.data[idx]

        # 转换为ID
        word_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        label_ids = [self.label2idx[l] for l in labels]

        # 创建双字符ID
        if self.create_bigram:
            bigram_ids = []
            for i in range(len(words)):
                if i < len(words) - 1:
                    bigram = words[i] + words[i + 1]
                    bigram_id = self.bigram2idx.get(bigram, self.bigram2idx['<UNK>'])
                else:
                    bigram_id = self.bigram2idx['<PAD>']
                bigram_ids.append(bigram_id)
        else:
            bigram_ids = [0] * len(words)

        # 截断或填充
        length = len(word_ids)
        if length > self.max_len:
            word_ids = word_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
            bigram_ids = bigram_ids[:self.max_len]
            length = self.max_len
        else:
            word_ids = word_ids + [0] * (self.max_len - length)
            label_ids = label_ids + [0] * (self.max_len - length)
            bigram_ids = bigram_ids + [0] * (self.max_len - length)

        # 创建掩码
        mask = [1] * length + [0] * (self.max_len - length)

        return {
            'word_ids': torch.LongTensor(word_ids),
            'label_ids': torch.LongTensor(label_ids),
            'bigram_ids': torch.LongTensor(bigram_ids),
            'mask': torch.FloatTensor(mask),
            'length': torch.LongTensor([length])
        }


def collate_fn_flat(batch):
    """FLAT的批处理函数"""
    word_ids = torch.stack([item['word_ids'] for item in batch])
    label_ids = torch.stack([item['label_ids'] for item in batch])
    bigram_ids = torch.stack([item['bigram_ids'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    lengths = torch.cat([item['length'] for item in batch])

    return word_ids, label_ids, bigram_ids, mask, lengths


def get_flat_dataloader(file_path, batch_size, shuffle=True, **kwargs):
    """获取FLAT数据加载器"""
    dataset = FLATDataset(file_path, **kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_flat,
        num_workers=0
    )
    return dataloader, dataset
