import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class LatticeLSTM(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, word_vocab_size,
                 embedding_dim, hidden_dim, num_labels, dropout=0.5):
        super(LatticeLSTM, self).__init__()

        # 字符和词嵌入
        self.char_embedding = nn.Embedding(char_vocab_size, embedding_dim)
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim)

        # Lattice LSTM层
        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 bidirectional=True, batch_first=True)
        self.word_lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 bidirectional=True, batch_first=True)

        # 门控机制
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)

    def forward(self, char_ids, word_ids, word_positions, attention_mask, labels=None):
        # 字符嵌入和LSTM
        char_embeds = self.char_embedding(char_ids)
        char_lstm_out, _ = self.char_lstm(char_embeds)

        # 词嵌入和LSTM（如果有词信息）
        if word_ids is not None:
            word_embeds = self.word_embedding(word_ids)
            word_lstm_out, _ = self.word_lstm(word_embeds)

            # 将词信息集成到相应位置
            batch_size, seq_len, hidden_dim = char_lstm_out.size()
            integrated_out = char_lstm_out.clone()

            for b in range(batch_size):
                for w_idx, (start, end) in enumerate(word_positions[b]):
                    if start < seq_len and end <= seq_len:
                        # 使用门控机制融合字符和词信息
                        char_hidden = char_lstm_out[b, start:end].mean(0)
                        word_hidden = word_lstm_out[b, w_idx]
                        gate_value = torch.sigmoid(self.gate(torch.cat([char_hidden, word_hidden], dim=-1)))
                        integrated_out[b, start:end] = gate_value * word_hidden + (1 - gate_value) * char_lstm_out[
                            b, start:end]

            lstm_out = integrated_out
        else:
            lstm_out = char_lstm_out

        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            # 计算CRF损失或使用交叉熵
            loss = nn.CrossEntropyLoss()(emissions.view(-1, emissions.size(-1)), labels.view(-1))
            return loss
        else:
            # 返回预测结果
            return torch.argmax(emissions, dim=-1)


class LatticeLSTMModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)
        self.word_vocab = self.build_word_vocab()

    def build_word_vocab(self):
        # 构建词汇表（简化版本）
        return {'<PAD>': 0, '<UNK>': 1}

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = LatticeLSTM(
            vocab_size=vocab_size,
            char_vocab_size=vocab_size,
            word_vocab_size=len(self.word_vocab),
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def prepare_lattice_input(self, batch):
        """准备Lattice LSTM的输入"""
        # 这里简化处理，实际需要词典匹配
        return {
            'char_ids': batch['input_ids'],
            'word_ids': None,  # 简化版本，暂不使用词信息
            'word_positions': None,
            'attention_mask': batch['attention_mask'],
            'labels': batch.get('labels')
        }

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training Lattice LSTM"):
            optimizer.zero_grad()

            lattice_input = self.prepare_lattice_input(batch)
            lattice_input = {k: v.to(self.device) if v is not None and hasattr(v, 'to') else v
                             for k, v in lattice_input.items()}

            loss = self.model(**lattice_input)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Lattice LSTM"):
                lattice_input = self.prepare_lattice_input(batch)
                lattice_input = {k: v.to(self.device) if v is not None and hasattr(v, 'to') else v
                                 for k, v in lattice_input.items() if k != 'labels'}

                original_labels = batch['original_labels']

                start_time = time.time()
                predictions = self.model(**lattice_input)
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
