from typing import List, Dict, Any, Set
import numpy as np
from collections import defaultdict


class EvaluationMetrics:
    def __init__(self):
        self.metrics_history = defaultdict(list)

    def precision_at_k(self, results: List[Dict], relevant_docs: Set[str], k: int) -> float:
        """Precision@K"""
        if k == 0:
            return 0.0

        retrieved = [result['doc_id'] for result in results[:k]]
        hits = sum(1 for doc_id in retrieved if doc_id in relevant_docs)
        return hits / k

    def recall_at_k(self, results: List[Dict], relevant_docs: Set[str], k: int) -> float:
        """Recall@K"""
        if len(relevant_docs) == 0:
            return 0.0

        retrieved = [result['doc_id'] for result in results[:k]]
        hits = sum(1 for doc_id in retrieved if doc_id in relevant_docs)
        return hits / len(relevant_docs)

    def f1_at_k(self, results: List[Dict], relevant_docs: Set[str], k: int) -> float:
        """F1@K"""
        precision = self.precision_at_k(results, relevant_docs, k)
        recall = self.recall_at_k(results, relevant_docs, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def average_precision(self, results: List[Dict], relevant_docs: Set[str]) -> float:
        """Average Precision (AP)"""
        precision_scores = []
        num_hits = 0

        for i, result in enumerate(results):
            if result['doc_id'] in relevant_docs:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                precision_scores.append(precision_at_i)

        if not precision_scores:
            return 0.0

        return sum(precision_scores) / len(relevant_docs)

    def mean_average_precision(self, all_results: Dict[str, List], relevance_data: Dict) -> float:
        """Mean Average Precision (MAP)"""
        ap_scores = []

        for query, results in all_results.items():
            relevant_docs = relevance_data.get(query, set())
            ap = self.average_precision(results, relevant_docs)
            ap_scores.append(ap)

        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

    def mean_reciprocal_rank(self, all_results: Dict[str, List], relevance_data: Dict) -> float:
        """Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []

        for query, results in all_results.items():
            relevant_docs = relevance_data.get(query, set())
            for i, result in enumerate(results):
                if result['doc_id'] in relevant_docs:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def ndcg_at_k(self, results: List[Dict], relevant_docs: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        if k == 0:
            return 0.0

        # 计算DCG
        dcg = 0.0
        for i in range(min(k, len(results))):
            rel = 1 if results[i]['doc_id'] in relevant_docs else 0
            dcg += rel / np.log2(i + 2)  # i+2 because i starts from 0

        # 计算IDCG
        ideal_relevance = [1] * min(len(relevant_docs), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        return dcg / idcg if idcg > 0 else 0.0

    def comprehensive_evaluation(self, results: List[Dict], relevant_docs: Set[str],
                                 k_values: List[int] = None) -> Dict[str, Any]:
        """综合评估"""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        evaluation = {}

        # 各种K值的评估
        for k in k_values:
            evaluation[f'precision@{k}'] = self.precision_at_k(results, relevant_docs, k)
            evaluation[f'recall@{k}'] = self.recall_at_k(results, relevant_docs, k)
            evaluation[f'f1@{k}'] = self.f1_at_k(results, relevant_docs, k)
            evaluation[f'ndcg@{k}'] = self.ndcg_at_k(results, relevant_docs, k)

        # 整体评估指标
        evaluation['ap'] = self.average_precision(results, relevant_docs)
        evaluation['num_retrieved'] = len(results)
        evaluation['num_relevant'] = len(relevant_docs)
        evaluation['num_relevant_retrieved'] = len(
            [r for r in results if r['doc_id'] in relevant_docs]
        )

        return evaluation

    def compare_retrievers(self, all_results: Dict[str, Dict[str, List]],
                           relevance_data: Dict) -> Dict[str, Any]:
        """比较多个检索器的性能"""
        comparison = {}

        for retriever_name, query_results in all_results.items():
            retriever_metrics = {}

            for k in [1, 3, 5, 10]:
                precisions = []
                recalls = []
                f1s = []

                for query, results in query_results.items():
                    relevant_docs = relevance_data.get(query, set())
                    eval_result = self.comprehensive_evaluation(results, relevant_docs, [k])

                    precisions.append(eval_result[f'precision@{k}'])
                    recalls.append(eval_result[f'recall@{k}'])
                    f1s.append(eval_result[f'f1@{k}'])

                retriever_metrics[f'avg_precision@{k}'] = np.mean(precisions) if precisions else 0
                retriever_metrics[f'avg_recall@{k}'] = np.mean(recalls) if recalls else 0
                retriever_metrics[f'avg_f1@{k}'] = np.mean(f1s) if f1s else 0

            comparison[retriever_name] = retriever_metrics

        return comparison