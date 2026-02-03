import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import List, Dict, Tuple
import time
from collections import defaultdict


class NERMetrics:
    def __init__(self, label_list: List[str]):
        self.label_list = label_list
        self.reset()

    def reset(self):
        self.predictions = []
        self.references = []
        self.inference_times = []

    def add_batch(self, predictions: List[List[str]], references: List[List[str]], batch_time: float = None):
        """添加一个批次的预测和真实标签"""
        # 验证每个句子的长度匹配
        assert len(predictions) == len(references), f"批次中句子数量不匹配: {len(predictions)} vs {len(references)}"

        for pred, ref in zip(predictions, references):
            assert len(pred) == len(ref), f"句子长度不匹配: {len(pred)} vs {len(ref)}"

        self.predictions.extend(predictions)
        self.references.extend(references)
        if batch_time:
            self.inference_times.append(batch_time)

    def compute(self) -> Dict:
        """计算所有指标"""
        # 确保有数据
        if not self.predictions or not self.references:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'classification_report': "No data to evaluate",
                'entity_metrics': {},
                'performance_metrics': {},
                'error_analysis': {}
            }

        # 手动计算指标，避免 seqeval 的兼容性问题
        results = self.compute_token_metrics()

        # 实体级别的指标
        entity_metrics = self.compute_entity_metrics()

        # 性能指标
        performance_metrics = self.compute_performance_metrics()

        # 错误分析
        error_analysis = self.analyze_errors()

        # 分类报告
        classification_report = self.generate_classification_report()

        return {
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'accuracy': results['accuracy'],
            'classification_report': classification_report,
            'entity_metrics': entity_metrics,
            'performance_metrics': performance_metrics,
            'error_analysis': error_analysis
        }

    def compute_token_metrics(self) -> Dict:
        """计算token级别的指标"""
        # 展平所有标签
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(self.references, self.predictions):
            # 确保长度相同
            min_len = min(len(true_seq), len(pred_seq))
            y_true_flat.extend(true_seq[:min_len])
            y_pred_flat.extend(pred_seq[:min_len])

        # 计算每个标签类型的指标
        labels = [l for l in self.label_list if l != 'O']  # 排除O标签

        # 整体准确率
        accuracy = accuracy_score(y_true_flat, y_pred_flat)

        # 计算非O标签的精确率、召回率和F1
        y_true_binary = [1 if label != 'O' else 0 for label in y_true_flat]
        y_pred_binary = [1 if label != 'O' else 0 for label in y_pred_flat]

        # 计算TP, FP, FN
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def compute_entity_metrics(self) -> Dict:
        """计算实体级别的指标"""
        entity_types = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for pred_seq, ref_seq in zip(self.predictions, self.references):
            # 确保长度相同
            min_len = min(len(pred_seq), len(ref_seq))
            pred_seq = pred_seq[:min_len]
            ref_seq = ref_seq[:min_len]

            pred_entities = self.extract_entities(pred_seq)
            ref_entities = self.extract_entities(ref_seq)

            for entity_type in set(e[2] for e in pred_entities + ref_entities):
                pred_set = {(e[0], e[1]) for e in pred_entities if e[2] == entity_type}
                ref_set = {(e[0], e[1]) for e in ref_entities if e[2] == entity_type}

                entity_types[entity_type]['tp'] += len(pred_set & ref_set)
                entity_types[entity_type]['fp'] += len(pred_set - ref_set)
                entity_types[entity_type]['fn'] += len(ref_set - pred_set)

        # 计算每个实体类型的指标
        results = {}
        for entity_type, counts in entity_types.items():
            precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
            recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': counts['tp'] + counts['fn']
            }

        return results

    def extract_entities(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        """提取实体及其位置"""
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                entity_type = labels[i][2:]
                start = i
                i += 1
                while i < len(labels) and labels[i] == f'I-{entity_type}':
                    i += 1
                entities.append((start, i, entity_type))
            else:
                i += 1
        return entities

    def compute_performance_metrics(self) -> Dict:
        """计算性能相关指标"""
        if self.inference_times:
            return {
                'avg_inference_time': np.mean(self.inference_times),
                'std_inference_time': np.std(self.inference_times),
                'total_inference_time': sum(self.inference_times),
                'samples_per_second': len(self.predictions) / sum(self.inference_times) if sum(
                    self.inference_times) > 0 else 0
            }
        return {}

    def analyze_errors(self) -> Dict:
        """错误分析"""
        error_types = defaultdict(int)

        for pred_seq, ref_seq in zip(self.predictions, self.references):
            min_len = min(len(pred_seq), len(ref_seq))
            for i in range(min_len):
                pred = pred_seq[i]
                ref = ref_seq[i]

                if pred != ref:
                    # 分类错误类型
                    if ref == 'O' and pred != 'O':
                        error_types['false_positive'] += 1
                    elif ref != 'O' and pred == 'O':
                        error_types['false_negative'] += 1
                    elif ref.startswith('B-') and pred.startswith('I-'):
                        error_types['boundary_error'] += 1
                    elif ref.startswith('I-') and pred.startswith('B-'):
                        error_types['boundary_error'] += 1
                    elif ref[2:] != pred[2:] and ref != 'O' and pred != 'O':
                        error_types['type_error'] += 1
                    else:
                        error_types['other'] += 1

        total_errors = sum(error_types.values())
        return {
            'total_errors': total_errors,
            'error_distribution': {k: v / total_errors if total_errors > 0 else 0 for k, v in error_types.items()},
            'error_counts': dict(error_types)
        }

    def generate_classification_report(self) -> str:
        """生成分类报告"""
        # 展平标签
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(self.references, self.predictions):
            min_len = min(len(true_seq), len(pred_seq))
            y_true_flat.extend(true_seq[:min_len])
            y_pred_flat.extend(pred_seq[:min_len])

        # 统计每个标签的指标
        label_metrics = {}

        for label in self.label_list:
            tp = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == label and p != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn

            label_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }

        # 格式化报告
        report = "              precision    recall  f1-score   support\n\n"

        for label, metrics in label_metrics.items():
            report += f"{label:>12}      {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}    {metrics['support']:>7}\n"

        # 计算平均值
        avg_precision = np.mean([m['precision'] for m in label_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in label_metrics.values()])
        avg_f1 = np.mean([m['f1-score'] for m in label_metrics.values()])
        total_support = sum(m['support'] for m in label_metrics.values())

        report += f"\n{'avg/total':>12}      {avg_precision:.2f}      {avg_recall:.2f}      {avg_f1:.2f}    {total_support:>7}\n"

        return report
