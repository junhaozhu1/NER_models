import torch

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据路径
TRAIN_FILE = 'data/msra/train.txt'
TEST_FILE = 'data/msra/test.txt'

# 模型配置
MODEL_CONFIGS = {
    'bilstm-crf': {
        'embedding_dim': 100,
        'hidden_dim': 256,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'use_amp': True,
        'gradient_clip': 5.0
    },
    'bert-crf': {
        'bert_model': 'bert-base-chinese',
        'dropout': 0.3,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 10,
        'use_amp': True,
        'gradient_clip': 1.0,
        'warmup_steps': 500
    },
    'can-ner': {
        'embedding_dim': 100,
        'hidden_dim': 256,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'use_amp': True,
        'gradient_clip': 5.0
    },
    'can-ner-bert': {
        'bert_model': 'bert-base-chinese',
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 10,
        'use_amp': True,
        'gradient_clip': 1.0,
        'warmup_steps': 500
    }
}

# 通用配置
MAX_LEN = 128
SEED = 42
