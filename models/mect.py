import torch
import torch.nn as nn
from .base_model import BaseNERModel
import time
from tqdm import tqdm


class MultiEncoderCrossTransformer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiEncoderCrossTransformer, self).__init__()

        self.self_attn_char = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.self_attn_word = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.cross_attn_c2w = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_w2c = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, char_features, word_features, mask=None):
        # 自注意力
        char_self_attn, _ = self.self_attn_char(char_features, char_features, char_features)
        char_features = self.norm1(char_features + self.dropout(char_self_attn))

        word_self_attn, _ = self.self_attn_word(word_features, word_features, word_features)
        word_features = self.norm2(word_features + self.dropout(word_self_attn))

        # 交叉注意力
        c2w_attn, _ = self.cross_attn_c2w(char_features, word_features, word_features)
        char_features = self.norm3(char_features + self.dropout(c2w_attn))

        w2c_attn, _ = self.cross_attn_w2c(word_features, char_features, char_features)
        word_features = self.norm4(word_features + self.dropout(w2c_attn))

        # FFN
        char_features = char_features + self.dropout(self.ffn(char_features))

        return char_features, word_features


class MECT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels,
                 n_layers=6, n_heads=8, dropout=0.1):
        super(MECT, self).__init__()

        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.char_encoder = nn.LSTM(embedding_dim, hidden_dim // 2,
                                    bidirectional=True, batch_first=True)
        self.word_encoder = nn.LSTM(embedding_dim, hidden_dim // 2,
                                    bidirectional=True, batch_first=True)

        self.mect_layers = nn.ModuleList([
            MultiEncoderCrossTransformer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 嵌入
        char_embeds = self.char_embedding(input_ids)
        word_embeds = self.word_embedding(input_ids)  # 简化

        # 初步编码
        char_features, _ = self.char_encoder(char_embeds)
        word_features, _ = self.word_encoder(word_embeds)

        # MECT层
        for mect_layer in self.mect_layers:
            char_features, word_features = mect_layer(char_features, word_features)

        # 分类
        output = self.dropout(char_features)
        logits = self.classifier(output)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        else:
            return torch.argmax(logits, dim=-1)


class MECTModel(BaseNERModel):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, vocab_size: int, num_labels: int):
        self.model = MECT(
            vocab_size=vocab_size,
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            num_labels=num_labels,
            dropout=self.config.DROPOUT
        )
        self.model.to(self.device)
        return self.model

    def train_epoch(self, train_loader, optimizer, criterion=None):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training MECT"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss = self.model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating MECT"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                original_labels = batch['original_labels']

                start_time = time.time()
                predictions = self.model(input_ids, attention_mask)
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
