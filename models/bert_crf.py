import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF


class BertCRF(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_labels=7, dropout=0.3):
        super().__init__()

        # Transformers 4.37.3 的新特性
        self.bert = BertModel.from_pretrained(
            bert_model_name,
            add_pooling_layer=False,  # NER不需要pooling层
            return_dict=True  # 返回字典格式
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # 初始化分类器权重
        self.classifier.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 使用 transformers 4.37.3 的输出格式
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        return emissions

    def loss(self, input_ids, label_ids, attention_mask):
        emissions = self.forward(input_ids, attention_mask)
        # CRF需要bool类型的mask
        mask = attention_mask.bool()
        return -self.crf(emissions, label_ids, mask=mask, reduction='mean')

    def predict(self, input_ids, attention_mask):
        emissions = self.forward(input_ids, attention_mask)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)
