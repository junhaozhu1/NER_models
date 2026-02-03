import os


class Config:
    # 数据路径
    DATA_DIR = "data/msra"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")

    # 模型参数
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    DROPOUT = 0.5

    # 词向量参数
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256

    # BERT参数
    BERT_MODEL_NAME = "bert-base-chinese"

    # 标签
    LABELS = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

    # 输出路径
    OUTPUT_DIR = "output"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "../models")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

    # 随机种子
    SEED = 42
