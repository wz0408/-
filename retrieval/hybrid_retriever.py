from .bm25_retriever import BM25Retriever
from .tfidf_retriever import TfidfRetriever
import numpy as np


class HybridRetriever:
    def __init__(self, methods=None, weights=None):
        if methods is None:
            methods = ['bm25', 'tfidf']

        self.methods = methods
        self.weights = weights or self._get_default_weights(methods)

        # 初始化各个检索器
        self.retrievers = {}
        self._initialize_retrievers()

    # 在hybrid_retriever.py中调整权重
    def _get_default_weights(self, methods):
        """默认权重配置 - 基于性能调整"""
        # 根据测试结果，BM25表现更好，给予更高权重
        weight_map = {'bm25': 0.7, 'tfidf': 0.3}

        weights = {}
        total_weight = 0

        for method in methods:
            weights[method] = weight_map.get(method, 0.5)
            total_weight += weights[method]

        # 归一化
        for method in weights:
            weights[method] /= total_weight

        return weights

    def _initialize_retrievers(self):
        """初始化检索器"""
        for method in self.methods:
            try:
                if method == 'bm25':
                    self.retrievers['bm25'] = BM25Retriever()
                elif method == 'tfidf':
                    self.retrievers['tfidf'] = TfidfRetriever()
                print(f"✓ 初始化{method}检索器")
            except Exception as e:
                print(f"✗✗ 初始化{method}检索器失败: {e}")

    def _normalize_scores(self, scores):
        """改进的分数归一化"""
        if not scores:
            return [0.1] * len(scores)  # 返回基础分数而不是0

        scores = np.array(scores)

        # 处理异常值
        scores = np.nan_to_num(scores, nan=0.1, posinf=1.0, neginf=0.1)

        min_score = np.min(scores)
        max_score = np.max(scores)

        # 如果所有分数相同，返回基础分数
        if max_score - min_score < 1e-10:
            if max_score == 0:
                return [0.5] * len(scores)  # 如果都是0，给中等分数
            else:
                return scores.tolist()  # 保持原分数

        # 归一化到 [0.1, 1.0] 范围
        normalized = 0.1 + 0.9 * (scores - min_score) / (max_score - min_score)
        return normalized.tolist()

    def retrieve(self, query, top_k=5, fusion_method='weighted'):
        """混合检索 - 改进版本"""
        all_results = {}

        # 从各个检索器获取结果
        for method, retriever in self.retrievers.items():
            try:
                results = retriever.retrieve(query, top_k * 2)  # 获取更多结果用于融合
                all_results[method] = results
            except Exception as e:
                print(f"检索器{method}执行失败: {e}")
                # 返回空结果而不是失败
                all_results[method] = []

        # 结果融合
        if fusion_method == 'weighted':
            return self._weighted_fusion(all_results, top_k)
        elif fusion_method == 'rrf':  # 新增：倒数排名融合
            return self._reciprocal_rank_fusion(all_results, top_k)
        else:
            return self._weighted_fusion(all_results, top_k)

    def _weighted_fusion(self, all_results, top_k):
        """加权融合 - 改进版本"""
        # 收集所有文档
        all_docs = set()
        for method, results in all_results.items():
            for result in results:
                all_docs.add(result['doc_id'])

        if not all_docs:
            return self._get_fallback_results(top_k)

        # 为每个文档计算加权分数
        doc_scores = {}
        doc_texts = {}

        # 获取第一个检索器来获取文档文本
        first_method = list(self.retrievers.keys())[0]
        if all_results.get(first_method):
            for result in all_results[first_method]:
                doc_texts[result['doc_id']] = result['text']

        # 为每个文档在每个检索器中计算分数
        for doc_id in all_docs:
            total_score = 0
            method_scores = {}

            for method, results in all_results.items():
                # 查找文档在该检索器中的分数
                score = 0.1  # 基础分数
                for result in results:
                    if result['doc_id'] == doc_id:
                        score = result['score']
                        break

                method_scores[method] = score
                weight = self.weights.get(method, 0.1)
                total_score += score * weight

            doc_scores[doc_id] = {
                'total_score': total_score,
                'method_scores': method_scores,
                'text': doc_texts.get(doc_id, f"文档 {doc_id} 的内容")
            }

        # 按总分排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)

        results = []
        for i, (doc_id, info) in enumerate(sorted_docs[:top_k]):
            results.append({
                'doc_id': doc_id,
                'text': info['text'],
                'score': info['total_score'],
                'retriever': 'hybrid_weighted',
                'component_scores': info['method_scores'],
                'rank': i + 1
            })

        return results

    def _reciprocal_rank_fusion(self, all_results, top_k, k=60):
        """倒数排名融合 (RRF) - 更先进的融合方法"""
        # 收集所有文档的RRF分数
        rrf_scores = {}
        doc_texts = {}

        # 获取文档文本
        for method, results in all_results.items():
            for result in results:
                doc_id = result['doc_id']
                if doc_id not in doc_texts:
                    doc_texts[doc_id] = result['text']

        # 计算每个文档的RRF分数
        for method, results in all_results.items():
            for rank, result in enumerate(results):
                doc_id = result['doc_id']
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        if not rrf_scores:
            return self._get_fallback_results(top_k)

        # 按RRF分数排序
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            # 归一化分数到 [0.1, 1.0]
            normalized_score = 0.1 + 0.9 * (score / max(rrf_scores.values()))

            results.append({
                'doc_id': doc_id,
                'text': doc_texts.get(doc_id, f"文档 {doc_id} 的内容"),
                'score': normalized_score,
                'retriever': 'hybrid_rrf',
                'rank': i + 1
            })

        return results

    def _get_fallback_results(self, top_k):
        """备选方案：使用BM25检索器"""
        if 'bm25' in self.retrievers:
            return self.retrievers['bm25'].retrieve("", top_k)
        else:
            # 返回空结果
            return []