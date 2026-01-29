import pickle
from collections import defaultdict


class Lexicon:
    """词典管理器，用于Lattice LSTM"""

    def __init__(self, word2idx=None):
        self.word2idx = word2idx or {'<PAD>': 0, '<UNK>': 1}
        self.trie = {}

    def add_word(self, word):
        """添加词到词典"""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)

        # 构建Trie树
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = self.word2idx[word]  # 词的结尾标记

    def load_lexicon(self, lexicon_path):
        """从文件加载词典"""
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    self.add_word(word)
        print(f"加载词典完成，共{len(self.word2idx)}个词")

    def match_sentence(self, sentence):
        """
        匹配句子中的所有词
        Returns:
            word_ids_list: {position: [(word_id, (start, end))]}
        """
        word_matches = defaultdict(list)

        for i in range(len(sentence)):
            node = self.trie
            for j in range(i, len(sentence)):
                char = sentence[j]
                if char not in node:
                    break
                node = node[char]

                # 如果找到完整的词
                if '#' in node:
                    word_id = node['#']
                    # 以结束位置为key存储
                    word_matches[j].append((word_id, (i, j + 1)))

        return word_matches

    def save(self, path):
        """保存词典"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'trie': self.trie
            }, f)

    def load(self, path):
        """加载词典"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.trie = data['trie']


def create_lexicon_from_training_data(train_file, min_freq=2):
    """从训练数据中提取词典"""
    word_freq = defaultdict(int)

    with open(train_file, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            line = line.strip()
            if not line:
                # 处理收集的词
                if len(words) > 1:  # 多字词
                    word = ''.join(words)
                    word_freq[word] += 1
                words = []
            else:
                char, label = line.split()
                if label.startswith('B-'):
                    if words and len(words) > 1:
                        word = ''.join(words)
                        word_freq[word] += 1
                    words = [char]
                elif label.startswith('I-') or label.startswith('M-') or label.startswith('E-'):
                    words.append(char)
                else:  # O标签
                    if words and len(words) > 1:
                        word = ''.join(words)
                        word_freq[word] += 1
                    words = []

    # 过滤低频词
    lexicon = Lexicon()
    for word, freq in word_freq.items():
        if freq >= min_freq:
            lexicon.add_word(word)

    return lexicon
