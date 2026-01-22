import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import argparse
import os
from datetime import datetime

# 导入自定义模块
from data.utils import get_dataloader
from models.bilstm_crf import BiLSTMCRF
from models.bert_crf import BertCRF
import config

warnings.filterwarnings('ignore')

def compute_loss(model, model_name, word_ids, label_ids, mask, lengths):
    """统一的loss计算接口"""
    if model_name == 'bert-crf':
        return model.loss(word_ids, label_ids, mask)
    else:  # bilstm-crf 或其他需要lengths的模型
        return model.loss(word_ids, label_ids, mask, lengths)
def predict_batch(model, model_name, word_ids, mask, lengths):
    """统一的预测接口"""
    if model_name == 'bert-crf':
        return model.predict(word_ids, mask)
    else:
        return model.predict(word_ids, mask, lengths)

def train_model(model_name='bilstm-crf'):
    """
    训练指定的NER模型

    Args:
        model_name: 模型名称，可选 'bilstm-crf', 'bert-crf' 等
    """
    # 获取配置
    cfg = config.MODEL_CONFIGS[model_name]
    device = config.DEVICE

    print(f"使用设备: {device}")
    print(f"训练模型: {model_name}")

    # 加载数据
    print("加载数据...")
    train_loader, train_dataset = get_dataloader(
        config.TRAIN_FILE,
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    test_loader, test_dataset = get_dataloader(
        config.TEST_FILE,
        batch_size=cfg['batch_size'],
        shuffle=False,
        word2idx=train_dataset.word2idx,
        label2idx=train_dataset.label2idx
    )

    print(f"词表大小: {len(train_dataset.word2idx)}")
    print(f"标签数量: {len(train_dataset.label2idx)}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型
    print(f"创建模型: {model_name}")
    if model_name == 'bilstm-crf':
        model = BiLSTMCRF(
            vocab_size=len(train_dataset.word2idx),
            num_labels=len(train_dataset.label2idx),
            embedding_dim=cfg['embedding_dim'],
            hidden_dim=cfg['hidden_dim'],
            dropout=cfg['dropout']
        )
    elif model_name == 'bert-crf':
        model = BertCRF(
            bert_model_name=cfg['bert_model'],
            num_labels=len(train_dataset.label2idx),
            dropout=cfg['dropout']
        )
        # raise NotImplementedError(f"模型 {model_name} 尚未实现")
    else:
        raise ValueError(f"未知的模型: {model_name}")

    # 将模型移到设备上
    model = model.to(device)

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 使用混合精度训练（如果有GPU）
    use_amp = device == 'cuda' and cfg.get('use_amp', True)
    scaler = GradScaler() if use_amp else None

    # 训练循环
    print("\n开始训练...")
    best_f1 = 0
    patience = 5
    no_improve = 0

    for epoch in range(cfg['epochs']):
        # 训练阶段
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                word_ids, label_ids, mask, lengths = batch

                # 将数据移到设备上
                word_ids = word_ids.to(device)
                label_ids = label_ids.to(device)
                mask = mask.to(device)
                lengths = lengths.to(device)

                # 梯度清零
                optimizer.zero_grad(set_to_none=True)

                # 前向传播和反向传播
                if use_amp:
                    # with autocast():
                    #     loss = model.loss(word_ids, label_ids, mask, lengths)
                    # scaler.scale(loss).backward()
                    with autocast():
                        loss = compute_loss(model, model_name, word_ids, label_ids, mask, lengths)
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if cfg.get('gradient_clip', None):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # loss = model.loss(word_ids, label_ids, mask, lengths)
                    # loss.backward()
                    loss = compute_loss(model, model_name, word_ids, label_ids, mask, lengths)
                    loss.backward()

                    # 梯度裁剪
                    if cfg.get('gradient_clip', None):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])

                    optimizer.step()

                # 更新统计信息
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # 验证阶段（每个epoch都验证）
        print("评估模型...")
        metrics = evaluate_model(model, test_loader, test_dataset, device)
        f1_score = metrics['f1']

        # 保存最佳模型
        if f1_score > best_f1:
            best_f1 = f1_score
            no_improve = 0

            # 保存模型
            save_path = f'{model_name}_best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'word2idx': train_dataset.word2idx,
                'label2idx': train_dataset.label2idx,
                'idx2label': train_dataset.idx2label,
                'config': cfg
            }, save_path)
            print(f'保存最佳模型到 {save_path}, F1: {best_f1:.4f}')
        else:
            no_improve += 1

        # 早停
        if no_improve >= patience:
            print(f'验证集F1分数已经{patience}个epoch没有提升，停止训练')
            break

    print(f"\n训练完成！最佳F1分数: {best_f1:.4f}")
    return model, train_dataset


def evaluate_model(model, model_name, test_loader, test_dataset, device):
    """
    评估模型性能

    Returns:
        dict: 包含precision, recall, f1等指标
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估中'):
            word_ids, label_ids, mask, lengths = batch
            word_ids = word_ids.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

            # 预测
            # predictions = model.predict(word_ids, mask, lengths)
            predictions = predict_batch(model, model_name, word_ids, mask, lengths)

            # 收集预测结果
            for i, length in enumerate(lengths):
                true_labels = label_ids[i][:length].tolist()
                pred_labels = predictions[i][:length]

                all_true.extend([test_dataset.idx2label[l] for l in true_labels])
                all_pred.extend([test_dataset.idx2label[l] for l in pred_labels])

    # 计算指标
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    # 过滤掉'O'标签，只计算实体标签的性能
    entity_true = []
    entity_pred = []
    for t, p in zip(all_true, all_pred):
        if t != 'O' or p != 'O':  # 至少有一个不是O
            entity_true.append(t)
            entity_pred.append(p)

    # 计算总体指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        entity_true, entity_pred, average='micro'
    )

    print(f"\n性能指标:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 打印详细报告
    print("\n详细分类报告:")
    print(classification_report(all_true, all_pred, digits=4))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练NER模型')
    parser.add_argument('--model', type=str, default='bilstm-crf',
                        choices=['bilstm-crf', 'bert-crf'],
                        help='选择要训练的模型')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件中的设置）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置文件中的设置）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件中的设置）')

    args = parser.parse_args()

    # 覆盖配置（如果提供了命令行参数）
    if args.epochs:
        config.MODEL_CONFIGS[args.model]['epochs'] = args.epochs
    if args.batch_size:
        config.MODEL_CONFIGS[args.model]['batch_size'] = args.batch_size
    if args.lr:
        config.MODEL_CONFIGS[args.model]['learning_rate'] = args.lr

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 开始训练
    train_model(args.model)


if __name__ == "__main__":
    main()
