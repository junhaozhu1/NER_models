import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import List, Dict, Tuple
import time
from collections import defaultdict
import seqeval.metrics


class NERMetrics:
    def __init__(self, label_list: List[str]):
        self.label_list = label_list
        self.reset()

    def reset(self):
        self.predictions = []
        self.references = []
        self.inference_times = []

    def add_batch(self, predictions: List[List[str]], references: List[List[str]], batch_time: float = None):
        self.predictions.extend(predictions)
        self.references.extend(references)
        if batch_time:
            self.inference_times.append(batch_time)

    def compute(self) -> Dict:
        # 基础指标
        precision = seqeval.metrics.precision_score(self.references, self.predictions)
        recall = seqeval.metrics.recall_score(self.references, self.predictions)
        f1 = seqeval.metrics.f1_score(self.references, self.predictions)
        accuracy = seqeval.metrics.accuracy_score(self.references, self.predictions)

        # 分类报告
        classification_report = seqeval.metrics.classification_report(
            self.references, self.predictions, digits=4
        )

        # 实体级别的指标
        entity_metrics = self.compute_entity_metrics()

        # 性能指标
        performance_metrics = self.compute_performance_metrics()

        # 错误分析
        error_analysis = self.analyze_errors()

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'classification_report': classification_report,
            'entity_metrics': entity_metrics,
            'performance_metrics': performance_metrics,
            'error_analysis': error_analysis
        }

    def compute_entity_metrics(self) -> Dict:
        """计算实体级别的指标"""
        entity_types = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for pred_seq, ref_seq in zip(self.predictions, self.references):
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
            for i, (pred, ref) in enumerate(zip(pred_seq, ref_seq)):
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
