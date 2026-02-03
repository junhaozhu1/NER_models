import os
import json
import torch
import numpy as np
import random
import argparse
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from configs.config import Config
from utils.data_loader import create_data_loader
from utils.metrics import NERMetrics
from utils.visualization import plot_results

# 导入所有模型
from models.bilstm_crf import BiLSTMCRFModel
from models.bert_crf import BERTCRFModel
from models.can_ner import CANNERModel
from models.lattice_lstm import LatticeLSTMModel
from models.w2ner import W2NERModel
from models.flat import FLATModel
from models.softlexicon import SoftLexiconModel
from models.lebert import LEBERTModel
from models.mect import MECTModel
from models.zen import ZENModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NERBenchmark:
    def __init__(self, config):
        self.config = config
        self.results = self.load_results()  # 加载已有结果

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        # 设置随机种子
        set_seed(config.SEED)

        # 模型注册表
        self.model_registry = {
            # 'BiLSTM-CRF': BiLSTMCRFModel,  #good
            # 'BERT-CRF': BERTCRFModel,
            # 'CAN-NER': CANNERModel,   #good
            # 'Lattice-LSTM': LatticeLSTMModel,   #good
            # 'W2NER': W2NERModel,
            # 'FLAT': FLATModel,
            'SoftLexicon': SoftLexiconModel,
            'LEBERT': LEBERTModel,
            'MECT': MECTModel,
            'ZEN': ZENModel
        }

    def load_results(self):
        """加载已有的结果"""
        results_file = os.path.join(self.config.RESULTS_DIR, 'all_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}

    def save_results(self):
        """保存所有结果"""
        results_file = os.path.join(self.config.RESULTS_DIR, 'all_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def prepare_data(self):
        """准备数据加载器"""
        if not hasattr(self, 'train_loader'):
            self.train_loader = create_data_loader(
                self.config.TRAIN_FILE, self.config,
                self.config.BATCH_SIZE, shuffle=True
            )
            self.test_loader = create_data_loader(
                self.config.TEST_FILE, self.config,
                self.config.BATCH_SIZE, shuffle=False
            )
            self.vocab_size = self.train_loader.dataset.vocab_size
            self.num_labels = len(self.config.LABELS)

    def train_model(self, model_name, model_wrapper, resume=False):
        """训练单个模型"""
        print(f"\n{'=' * 50}")
        print(f"处理模型: {model_name}")
        print(f"{'=' * 50}")

        # 检查是否已有结果
        if model_name in self.results and not resume:
            print(f"模型 {model_name} 已有结果，跳过训练。使用 --resume 强制重新训练。")
            return self.results[model_name]

        # 构建模型
        model = model_wrapper.build_model(self.vocab_size, self.num_labels)

        # 检查是否有保存的模型
        model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"{model_name}_best.pth")
        start_epoch = 0
        best_f1 = 0

        if os.path.exists(model_path) and not resume:
            print(f"加载已保存的模型: {model_path}")
            model_wrapper.load_model(model_path)
            # 直接评估
            metrics = NERMetrics(self.config.LABELS)
            eval_results = model_wrapper.evaluate(self.test_loader, metrics)
            print(f"加载的模型 F1: {eval_results['f1']:.4f}")
            return eval_results

        # 训练模型
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        metrics = NERMetrics(self.config.LABELS)

        for epoch in range(start_epoch, self.config.EPOCHS):
            # 训练
            train_loss = model_wrapper.train_epoch(self.train_loader, optimizer)

            # 评估
            eval_results = model_wrapper.evaluate(self.test_loader, metrics)

            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Eval F1: {eval_results['f1']:.4f}")

            # 保存最佳模型
            if eval_results['f1'] > best_f1:
                best_f1 = eval_results['f1']
                model_wrapper.save_model(model_path)
                print(f"保存最佳模型，F1: {best_f1:.4f}")

        return eval_results

    def evaluate_only(self, model_names=None):
        """仅评估模型（不训练）"""
        self.prepare_data()

        if model_names is None:
            model_names = list(self.model_registry.keys())

        for model_name in model_names:
            if model_name not in self.model_registry:
                print(f"未知模型: {model_name}")
                continue

            model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"{model_name}_best.pth")
            if not os.path.exists(model_path):
                print(f"模型 {model_name} 未找到已保存的权重，跳过")
                continue

            print(f"\n评估模型: {model_name}")
            model_wrapper = self.model_registry[model_name](self.config)
            model_wrapper.build_model(self.vocab_size, self.num_labels)
            model_wrapper.load_model(model_path)

            metrics = NERMetrics(self.config.LABELS)
            eval_results = model_wrapper.evaluate(self.test_loader, metrics)

            self.results[model_name] = eval_results
            print(f"{model_name} - F1: {eval_results['f1']:.4f}")

        self.save_results()
        self.display_results()

    def run_benchmark(self, model_names=None, resume=False):
        """运行基准测试"""
        self.prepare_data()

        if model_names is None:
            model_names = list(self.model_registry.keys())

        for model_name in model_names:
            if model_name not in self.model_registry:
                print(f"未知模型: {model_name}")
                continue

            # 创建模型实例
            model_wrapper = self.model_registry[model_name](self.config)

            # 训练和评估
            eval_results = self.train_model(model_name, model_wrapper, resume)

            # 保存结果
            self.results[model_name] = eval_results
            self.save_results()

            # 保存详细结果
            with open(os.path.join(self.config.RESULTS_DIR, f"{model_name}_results.json"), 'w') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)

        # 显示结果
        self.display_results()

        # 生成可视化
        if len(self.results) > 1:
            self.visualize_results()

        # 保存报告
        self.save_report()

    def display_results(self):
        """显示结果表格"""
        if not self.results:
            print("没有可显示的结果")
            return

        results_list = []
        for model_name, eval_results in self.results.items():
            results_list.append({
                'Model': model_name,
                'Precision': f"{eval_results['precision']:.4f}",
                'Recall': f"{eval_results['recall']:.4f}",
                'F1-Score': f"{eval_results['f1']:.4f}",
                'Accuracy': f"{eval_results['accuracy']:.4f}",
                'Avg Inference Time (ms)': f"{eval_results.get('performance_metrics', {}).get('avg_inference_time', 0) * 1000:.2f}",
                'Samples/Second': f"{eval_results.get('performance_metrics', {}).get('samples_per_second', 0):.2f}"
            })

        print("\n" + "=" * 100)
        print("NER模型基准测试结果")
        print("=" * 100)

        df = pd.DataFrame(results_list)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        # 保存为CSV
        df.to_csv(os.path.join(self.config.RESULTS_DIR, 'benchmark_results.csv'), index=False)

    def visualize_results(self):
        """生成可视化图表"""
        # 实现保持不变
        pass

    def save_report(self):
        """生成并保存综合报告"""
        # 实现保持不变
        pass


def main():
    parser = argparse.ArgumentParser(description='NER模型基准测试')
    parser.add_argument('--models', nargs='+', help='要测试的模型名称列表')
    parser.add_argument('--evaluate-only', action='store_true', help='仅评估已训练的模型')
    parser.add_argument('--resume', action='store_true', help='强制重新训练已有结果的模型')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    parser.add_argument('--show-results', action='store_true', help='仅显示已有结果')

    args = parser.parse_args()

    # 创建配置
    config = Config()

    # 更新配置
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate

    # 创建基准测试实例
    benchmark = NERBenchmark(config)

    if args.show_results:
        # 仅显示已有结果
        benchmark.display_results()
    elif args.evaluate_only:
        # 仅评估模式
        benchmark.evaluate_only(args.models)
    else:
        # 训练和评估模式
        benchmark.run_benchmark(args.models, args.resume)


if __name__ == "__main__":
    main()
