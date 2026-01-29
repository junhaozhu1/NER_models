import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import math


class WordLSTMCell(nn.Module):
    """Lattice LSTM中的Word Cell"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 词的门控机制
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_f = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_c = nn.Linear(input_dim, hidden_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.W_i, self.W_f, self.W_c]:
            nn.init.xavier_uniform_(weight.weight)
            nn.init.zeros_(weight.bias)

    def forward(self, x_t):
        """
        Args:
            x_t: 词嵌入 [batch, word_dim]
        Returns:
            i_t, f_t, c_t: 门控值和细胞状态
        """
        i_t = torch.sigmoid(self.W_i(x_t))
        f_t = torch.sigmoid(self.W_f(x_t))
        c_tilde = torch.tanh(self.W_c(x_t))
        c_t = i_t * c_tilde

        return i_t, f_t, c_t


class LatticeLSTMCell(nn.Module):
    """Lattice LSTM的核心单元"""

    def __init__(self, char_dim, word_dim, hidden_dim):
        super().__init__()
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        # 字符LSTM参数
        self.W_char = nn.Linear(char_dim + hidden_dim, 4 * hidden_dim)

        # 词LSTM参数
        self.word_cell = WordLSTMCell(word_dim, hidden_dim)

        # 额外的门控，用于控制词信息
        self.W_word_gate = nn.Linear(hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_char.weight)
        nn.init.zeros_(self.W_char.bias)
        nn.init.xavier_uniform_(self.W_word_gate.weight)
        nn.init.zeros_(self.W_word_gate.bias)

    def forward(self, char_input, h_prev, c_prev, word_inputs=None, word_positions=None):
        """
        Args:
            char_input: 字符嵌入 [batch, char_dim]
            h_prev: 上一时刻隐状态 [batch, hidden_dim]
            c_prev: 上一时刻细胞状态 [batch, hidden_dim]
            word_inputs: 匹配到的词嵌入列表
            word_positions: 词的起始位置列表
        Returns:
            h_t: 当前隐状态
            c_t: 当前细胞状态
        """
        # 字符LSTM计算
        lstm_input = torch.cat([char_input, h_prev], dim=-1)
        gates = self.W_char(lstm_input)
        i_char, f_char, o_char, c_char = gates.chunk(4, dim=-1)

        i_char = torch.sigmoid(i_char)
        f_char = torch.sigmoid(f_char)
        o_char = torch.sigmoid(o_char)
        c_char = torch.tanh(c_char)

        # 基础细胞状态（只考虑字符）
        c_t = f_char * c_prev + i_char * c_char

        # 如果有词信息，融合词的贡献
        if word_inputs is not None and len(word_inputs) > 0:
            word_c_sum = torch.zeros_like(c_t)
            word_gate_sum = torch.zeros_like(c_t)

            for word_emb, (start_pos, end_pos) in zip(word_inputs, word_positions):
                # 计算词的贡献
                i_word, f_word, c_word = self.word_cell(word_emb)

                # 词的门控权重（考虑词的长度和位置）
                word_len = end_pos - start_pos
                alpha = 1.0 / (1.0 + word_len)  # 长词权重降低

                # 累加词的贡献
                word_c_sum += alpha * i_word * c_word
                word_gate_sum += alpha * i_word

            # 归一化并融合
            if word_gate_sum.sum() > 0:
                word_gate = torch.sigmoid(self.W_word_gate(word_gate_sum))
                c_t = (1 - word_gate) * c_t + word_gate * word_c_sum

        # 输出门
        h_t = o_char * torch.tanh(c_t)

        return h_t, c_t


class LatticeLSTM(nn.Module):
    """完整的Lattice LSTM模型"""

    def __init__(self, char_vocab_size, word_vocab_size, num_labels,
                 char_dim=100, word_dim=100, hidden_dim=256, dropout=0.5):
        super().__init__()

        self.char_dim = char_dim
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=0)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 前向和后向Lattice LSTM
        self.forward_cell = LatticeLSTMCell(char_dim, word_dim, hidden_dim)
        self.backward_cell = LatticeLSTMCell(char_dim, word_dim, hidden_dim)

        # 输出层
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.char_embedding.weight)
        nn.init.xavier_uniform_(self.word_embedding.weight)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

    def forward_lattice_lstm(self, char_embeds, word_ids_list, word_positions_list,
                             lengths, forward=True):
        """
        单向Lattice LSTM的前向传播

        Args:
            char_embeds: 字符嵌入 [batch, seq_len, char_dim]
            word_ids_list: 每个位置匹配到的词ID列表
            word_positions_list: 词的位置信息
            lengths: 序列长度
            forward: 是否是前向LSTM
        """
        batch_size, seq_len, _ = char_embeds.shape
        device = char_embeds.device

        # 初始化隐状态
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs = []

        # 选择处理顺序
        if forward:
            positions = range(seq_len)
            cell = self.forward_cell
        else:
            positions = range(seq_len - 1, -1, -1)
            cell = self.backward_cell

        for pos in positions:
            char_input = char_embeds[:, pos]

            # 获取当前位置的词信息
            batch_word_inputs = []
            batch_word_positions = []

            for batch_idx in range(batch_size):
                if pos >= lengths[batch_idx]:
                    batch_word_inputs.append([])
                    batch_word_positions.append([])
                    continue

                word_inputs = []
                word_positions = []

                # 收集以当前位置结尾的所有词
                if word_ids_list[batch_idx] and pos in word_ids_list[batch_idx]:
                    for word_id, (start, end) in word_ids_list[batch_idx][pos]:
                        if word_id > 0:  # 忽略padding
                            word_emb = self.word_embedding(
                                torch.tensor([word_id], device=device)
                            )
                            word_inputs.append(word_emb)
                            word_positions.append((start, end))

                batch_word_inputs.append(word_inputs)
                batch_word_positions.append(word_positions)

            # 更新隐状态
            new_h = []
            new_c = []

            for batch_idx in range(batch_size):
                if pos < lengths[batch_idx]:
                    h_i, c_i = cell(
                        char_input[batch_idx:batch_idx + 1],
                        h[batch_idx:batch_idx + 1],
                        c[batch_idx:batch_idx + 1],
                        batch_word_inputs[batch_idx],
                        batch_word_positions[batch_idx]
                    )
                    new_h.append(h_i)
                    new_c.append(c_i)
                else:
                    new_h.append(h[batch_idx:batch_idx + 1])
                    new_c.append(c[batch_idx:batch_idx + 1])

            h = torch.cat(new_h, dim=0)
            c = torch.cat(new_c, dim=0)
            outputs.append(h)

        # 调整输出顺序
        if not forward:
            outputs = outputs[::-1]

        return torch.stack(outputs, dim=1)

    def forward(self, char_ids, word_ids_list, word_positions_list, lengths, mask=None):
        """
        Args:
            char_ids: 字符ID [batch, seq_len]
            word_ids_list: 匹配到的词信息
            word_positions_list: 词位置信息
            lengths: 实际长度
            mask: 掩码
        """
        # 字符嵌入
        char_embeds = self.dropout(self.char_embedding(char_ids))

        # 前向Lattice LSTM
        forward_out = self.forward_lattice_lstm(
            char_embeds, word_ids_list, word_positions_list, lengths, forward=True
        )

        # 后向Lattice LSTM
        backward_out = self.forward_lattice_lstm(
            char_embeds, word_ids_list, word_positions_list, lengths, forward=False
        )

        # 拼接双向输出
        lstm_out = torch.cat([forward_out, backward_out], dim=-1)
        lstm_out = self.dropout(lstm_out)

        # 计算发射分数
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def loss(self, char_ids, label_ids, word_ids_list, word_positions_list, lengths, mask):
        """计算CRF损失"""
        emissions = self.forward(char_ids, word_ids_list, word_positions_list, lengths, mask)
        return -self.crf(emissions, label_ids, mask=mask, reduction='mean')

    def predict(self, char_ids, word_ids_list, word_positions_list, lengths, mask):
        """预测标签序列"""
        emissions = self.forward(char_ids, word_ids_list, word_positions_list, lengths, mask)
        return self.crf.decode(emissions, mask=mask)
