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
    },
    'w2ner': {
        'bert_model': 'bert-base-chinese',
        'dist_emb_dim': 64,
        'hidden_size': 768,
        'conv_channels': 128,
        'num_heads': 8,
        'dropout': 0.3,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 20,
        'use_amp': True,
        'gradient_clip': 1.0,
        'warmup_steps': 500
    },
    'lattice-lstm': {
        'char_dim': 100,
        'word_dim': 100,
        'hidden_dim': 256,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'batch_size': 16,  # Lattice LSTM计算量较大，使用较小的batch
        'epochs': 30,
        'use_amp': False,  # Lattice LSTM不建议使用混合精度
        'gradient_clip': 5.0,
        'lexicon_path': 'data/lexicon.pkl'  # 词典路径
    },
    'flat': {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 2,
        'd_ff': 1024,
        'dropout': 0.3,
        'use_bigram': True,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'use_amp': True,
        'gradient_clip': 5.0,
        'warmup_steps': 1000
    },

    'flat-lexicon': {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 1024,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 30,
        'use_amp': True,
        'gradient_clip': 5.0,
        'warmup_steps': 2000
    }
}

# 通用配置
MAX_LEN = 128
SEED = 42
