import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from .lexicon import Lexicon


class NERDataset(Dataset):
    """基础NER数据集（用于BiLSTM-CRF等）"""

    def __init__(self, file_path, word2idx=None, label2idx=None, max_len=128):
        self.max_len = max_len

        # 构建词表
        if word2idx is None:
            self.word2idx = self._build_vocab(file_path)
        else:
            self.word2idx = word2idx

        # 构建标签表
        if label2idx is None:
            self.label2idx = self._build_label_vocab(file_path)
        else:
            self.label2idx = label2idx

        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # 读取数据
        self.data = self._read_data(file_path)

    def _build_vocab(self, file_path):
        """构建字符词表"""
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    char, _ = line.split()
                    if char not in word2idx:
                        word2idx[char] = len(word2idx)
        return word2idx

    def _build_label_vocab(self, file_path):
        """构建标签词表"""
        label2idx = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    _, label = line.split()
                    if label not in label2idx:
                        label2idx[label] = len(label2idx)
        return label2idx

    def _read_data(self, file_path):
        """读取数据"""
        data = []
        chars, labels = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if chars:
                        data.append({
                            'chars': chars[:self.max_len],
                            'labels': labels[:self.max_len]
                        })
                        chars, labels = [], []
                else:
                    char, label = line.split()
                    chars.append(char)
                    labels.append(label)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chars = item['chars']
        labels = item['labels']

        # 转换为ID
        char_ids = [self.word2idx.get(c, self.word2idx['<UNK>']) for c in chars]
        label_ids = [self.label2idx[l] for l in labels]

        return {
            'char_ids': char_ids,
            'label_ids': label_ids,
            'length': len(char_ids)
        }


class LatticeNERDataset(NERDataset):
    """Lattice LSTM专用数据集（继承自基础数据集）"""

    def __init__(self, file_path, word2idx=None, label2idx=None, lexicon=None, max_len=128):
        super().__init__(file_path, word2idx, label2idx, max_len)
        self.lexicon = lexicon

        # 重新读取数据以包含词匹配信息
        if lexicon is not None:
            self.data = self._read_data_with_lexicon(file_path)

    def _read_data_with_lexicon(self, file_path):
        """读取数据并匹配词典"""
        data = []
        chars, labels = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if chars:
                        # 匹配词典中的词
                        word_matches = self.lexicon.match_sentence(chars)

                        data.append({
                            'chars': chars[:self.max_len],
                            'labels': labels[:self.max_len],
                            'word_matches': word_matches
                        })
                        chars, labels = [], []
                else:
                    char, label = line.split()
                    chars.append(char)
                    labels.append(label)

        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        chars = item['chars']
        labels = item['labels']
        word_matches = item.get('word_matches', {})

        # 转换为ID
        char_ids = [self.word2idx.get(c, self.word2idx['<UNK>']) for c in chars]
        label_ids = [self.label2idx[l] for l in labels]

        return {
            'char_ids': char_ids,
            'label_ids': label_ids,
            'word_matches': word_matches,
            'length': len(char_ids)
        }


def collate_fn_base(batch):
    """基础模型的批处理函数（BiLSTM-CRF等）"""
    # 获取批次数据
    char_ids = [torch.LongTensor(item['char_ids']) for item in batch]
    label_ids = [torch.LongTensor(item['label_ids']) for item in batch]
    lengths = torch.LongTensor([item['length'] for item in batch])

    # Padding
    char_ids_padded = pad_sequence(char_ids, batch_first=True, padding_value=0)
    label_ids_padded = pad_sequence(label_ids, batch_first=True, padding_value=0)

    # 创建mask
    batch_size, max_len = char_ids_padded.shape
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

    return char_ids_padded, label_ids_padded, mask, lengths


def collate_fn_lattice(batch):
    """Lattice LSTM的批处理函数"""
    # 获取批次数据
    char_ids = [item['char_ids'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    word_matches_list = [item['word_matches'] for item in batch]
    lengths = [item['length'] for item in batch]

    # 获取最大长度
    max_len = max(lengths)

    # Padding
    padded_char_ids = []
    padded_label_ids = []
    mask = []

    for i in range(len(batch)):
        # Pad sequences
        char_seq = char_ids[i] + [0] * (max_len - lengths[i])
        label_seq = label_ids[i] + [0] * (max_len - lengths[i])
        mask_seq = [1] * lengths[i] + [0] * (max_len - lengths[i])

        padded_char_ids.append(char_seq)
        padded_label_ids.append(label_seq)
        mask.append(mask_seq)

    # 转换为tensor
    char_ids_tensor = torch.LongTensor(padded_char_ids)
    label_ids_tensor = torch.LongTensor(padded_label_ids)
    mask_tensor = torch.BoolTensor(mask)
    lengths_tensor = torch.LongTensor(lengths)

    return {
        'char_ids': char_ids_tensor,
        'label_ids': label_ids_tensor,
        'word_ids_list': word_matches_list,
        'word_positions_list': word_matches_list,
        'mask': mask_tensor,
        'lengths': lengths_tensor
    }


def get_dataloader(file_path, batch_size=32, shuffle=True,
                   word2idx=None, label2idx=None, max_len=128):
    """获取基础数据加载器（用于BiLSTM-CRF等）"""
    dataset = NERDataset(file_path, word2idx, label2idx, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_base,
        num_workers=0  # Windows下设为0
    )
    return dataloader, dataset


def get_lattice_dataloader(file_path, lexicon, batch_size=32, shuffle=True,
                           word2idx=None, label2idx=None, max_len=128):
    """获取Lattice LSTM的数据加载器"""
    dataset = LatticeNERDataset(file_path, word2idx, label2idx, lexicon, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_lattice,
        num_workers=0
    )
    return dataloader, dataset


# BERT模型的数据处理（如果需要）
class BertNERDataset(Dataset):
    """BERT模型专用数据集"""

    def __init__(self, file_path, tokenizer, label2idx=None, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 构建标签表
        if label2idx is None:
            self.label2idx = self._build_label_vocab(file_path)
        else:
            self.label2idx = label2idx

        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # 读取数据
        self.data = self._read_data(file_path)

    def _build_label_vocab(self, file_path):
        """构建标签词表"""
        label2idx = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    _, label = line.split()
                    if label not in label2idx:
                        label2idx[label] = len(label2idx)
        return label2idx

    def _read_data(self, file_path):
        """读取数据"""
        data = []
        chars, labels = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if chars:
                        data.append({
                            'chars': chars,
                            'labels': labels
                        })
                        chars, labels = [], []
                else:
                    char, label = line.split()
                    chars.append(char)
                    labels.append(label)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chars = item['chars']
        labels = item['labels']

        # BERT tokenization
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # 对齐标签
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 特殊token
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2idx[labels[word_idx]])
            else:
                label_ids.append(-100)  # 子词token
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.LongTensor(label_ids)
        }


def get_bert_dataloader(file_path, tokenizer, batch_size=32, shuffle=True,
                        label2idx=None, max_len=128):
    """获取BERT模型的数据加载器"""
    dataset = BertNERDataset(file_path, tokenizer, label2idx, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return dataloader, dataset
