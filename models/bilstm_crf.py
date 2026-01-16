import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim=100, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # PyTorch 2.x 推荐使用 proj_size 来减少参数量
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # 使用新的初始化方法
        self._init_weights()

    def _init_weights(self):
        # PyTorch 2.x 推荐的初始化方式
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, word_ids, mask=None, lengths=None):
        embeddings = self.dropout(self.embedding(word_ids))

        # 使用 pack_padded_sequence 来提高效率（PyTorch 2.x优化）
        if lengths is not None:
            packed_embeddings = pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_embeddings)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(embeddings)

        emissions = self.hidden2tag(lstm_out)
        return emissions

    @torch.cuda.amp.autocast()  # PyTorch 2.x 自动混合精度
    def loss(self, word_ids, label_ids, mask, lengths=None):
        emissions = self.forward(word_ids, mask, lengths)
        return -self.crf(emissions, label_ids, mask=mask)

    def predict(self, word_ids, mask, lengths=None):
        emissions = self.forward(word_ids, mask, lengths)
        return self.crf.decode(emissions, mask=mask)
