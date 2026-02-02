import os
import json
import torch
import numpy as np
import random
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import Config
from utils.data_loader import create_data_loader
from utils.metrics import NERMetrics
# from utils.visualization import plot_results

# 导入所有模型
from models.bilstm_crf import BiLSTMCRFModel


# from models.bert_crf import BERTCRFModel
# ... 其他模型导入

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NERBenchmark:
    def __init__(self, config):
        self.config = config
        self.results = {}

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # 设置随机种子
        set_seed(config.SEED)

        # 加载数据
        self.train_loader = create_data_loader(
            config.TRAIN_FILE, config, config.BATCH_SIZE, shuffle=True
        )
        self.test_loader = create_data_loader(
            config.TEST_FILE, config, config.BATCH_SIZE, shuffle=False
        )

        # 获取词汇表大小和标签数量
        self.vocab_size = self.train_loader.dataset.vocab_size
        self.num_labels = len(config.LABELS)

        # 初始化模型
        self.models = {
            'BiLSTM-CRF': BiLSTMCRFModel(config),
            # 'BERT-CRF': BERTCRFModel(config),
            # 'CAN-NER': CANNERModel(config),
            # ... 添加其他模型
        }

    def train_model(self, model_name, model_wrapper):
        print(f"\n{'=' * 50}")
        print(f"训练模型: {model_name}")
        print(f"{'=' * 50}")

        # 构建模型
        model = model_wrapper.build_model(self.vocab_size, self.num_labels)

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

        # 训练
        best_f1 = 0
        metrics = NERMetrics(self.config.LABELS)

        for epoch in range(self.config.EPOCHS):
            # 训练
            train_loss = model_wrapper.train_epoch(self.train_loader, optimizer)

            # 评估
            eval_results = model_wrapper.evaluate(self.test_loader, metrics)

            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Eval Precision: {eval_results['precision']:.4f}")
            print(f"Eval Recall: {eval_results['recall']:.4f}")
            print(f"Eval F1: {eval_results['f1']:.4f}")
            print(f"Eval Accuracy: {eval_results['accuracy']:.4f}")

            # 保存最佳模型
            if eval_results['f1'] > best_f1:
                best_f1 = eval_results['f1']
                model_wrapper.save_model(
                    os.path.join(self.config.MODEL_SAVE_DIR, f"{model_name}_best.pth")
                )

        return eval_results

    def run_benchmark(self):
        """运行完整的基准测试"""
        results_list = []

        for model_name, model_wrapper in self.models.items():
            # 训练和评估模型
            eval_results = self.train_model(model_name, model_wrapper)

            # 保存结果
            self.results[model_name] = eval_results

            # 整理结果用于表格显示
            results_list.append({
                'Model': model_name,
                'Precision': f"{eval_results['precision']:.4f}",
                'Recall': f"{eval_results['recall']:.4f}",
                'F1-Score': f"{eval_results['f1']:.4f}",
                'Accuracy': f"{eval_results['accuracy']:.4f}",
                'Avg Inference Time (ms)': f"{eval_results['performance_metrics'].get('avg_inference_time', 0) * 1000:.2f}",
                'Samples/Second': f"{eval_results['performance_metrics'].get('samples_per_second', 0):.2f}"
            })

            # 保存详细结果
            with open(os.path.join(self.config.RESULTS_DIR, f"{model_name}_results.json"), 'w') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)

        # 显示结果表格
        self.display_results(results_list)

        # 生成可视化图表
        self.visualize_results()

        # 保存综合报告
        self.save_report(results_list)

    def display_results(self, results_list):
        """显示结果表格"""
        print("\n" + "=" * 100)
        print("NER模型基准测试结果")
        print("=" * 100)

        df = pd.DataFrame(results_list)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        # 保存为CSV
        df.to_csv(os.path.join(self.config.RESULTS_DIR, 'benchmark_results.csv'), index=False)

    def visualize_results(self):
        """生成可视化图表"""
        # 1. 主要指标对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        models = list(self.results.keys())
        metrics_data = {
            'Precision': [self.results[m]['precision'] for m in models],
            'Recall': [self.results[m]['recall'] for m in models],
            'F1-Score': [self.results[m]['f1'] for m in models],
            'Accuracy': [self.results[m]['accuracy'] for m in models]
        }

        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(models, values)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.set_xticklabels(models, rotation=45, ha='right')

            # 在柱子上添加数值
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'metrics_comparison.png'))

        # 2. 实体类型性能热图
        entity_metrics = {}
        for model_name, results in self.results.items():
            entity_metrics[model_name] = results['entity_metrics']

        # 创建热图数据
        entity_types = set()
        for model_metrics in entity_metrics.values():
            entity_types.update(model_metrics.keys())

        heatmap_data = []
        for entity_type in entity_types:
            row = []
            for model in models:
                if entity_type in entity_metrics[model]:
                    row.append(entity_metrics[model][entity_type]['f1'])
                else:
                    row.append(0)
            heatmap_data.append(row)

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                    xticklabels=models,
                    yticklabels=list(entity_types),
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd')
        plt.title('Entity-wise F1 Score Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'entity_performance_heatmap.png'))

    def save_report(self, results_list):
        """生成并保存综合报告"""
        report = f"""
# NER模型基准测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试配置
- 数据集: MSRA NER
- 训练样本数: {len(self.train_loader.dataset)}
- 测试样本数: {len(self.test_loader.dataset)}
- 批次大小: {self.config.BATCH_SIZE}
- 训练轮数: {self.config.EPOCHS}
- 学习率: {self.config.LEARNING_RATE}

## 总体性能对比

{tabulate(results_list, headers='keys', tablefmt='github')}

## 详细分析

### 1. 最佳模型
"""
        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        report += f"- 最高F1分数: {best_model[0]} ({best_model[1]['f1']:.4f})\n"

        # 添加错误分析
        report += "\n### 2. 错误分析\n\n"
        for model_name, results in self.results.items():
            error_analysis = results['error_analysis']
            report += f"**{model_name}**\n"
            report += f"- 总错误数: {error_analysis['total_errors']}\n"
            report += f"- 错误分布: {json.dumps(error_analysis['error_distribution'], indent=2)}\n\n"

        # 保存报告
        with open(os.path.join(self.config.RESULTS_DIR, 'benchmark_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)


if __name__ == "__main__":
    config = Config()
    benchmark = NERBenchmark(config)
    benchmark.run_benchmark()
