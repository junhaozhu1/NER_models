import torch
from data.utils import get_dataloader
from models.bilstm_crf import BiLSTMCRF
import config
from train import evaluate_model


def load_and_evaluate(model_name='bilstm-crf'):
    # 加载保存的模型
    checkpoint = torch.load(f'{model_name}_model.pth')

    # 创建数据加载器
    test_loader, test_dataset = get_dataloader(
        config.TEST_FILE,
        batch_size=64,
        shuffle=False,
        word2idx=checkpoint['word2idx'],
        label2idx=checkpoint['label2idx']
    )

    # 创建模型
    cfg = config.MODEL_CONFIGS[model_name]
    model = BiLSTMCRF(
        vocab_size=len(checkpoint['word2idx']),
        num_labels=len(checkpoint['label2idx']),
        embedding_dim=cfg['embedding_dim'],
        hidden_dim=cfg['hidden_dim']
    )

    model.load_state_dict(checkpoint['model'])
    model.to(config.DEVICE)

    # 评估
    evaluate_model(model, test_loader, test_dataset)


if __name__ == "__main__":
    load_and_evaluate('bilstm-crf')
