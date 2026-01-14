# evaluation/diagnostic_evaluator.py
import numpy as np
from collections import defaultdict
from .metrics import EvaluationMetrics


class DiagnosticEvaluator(EvaluationMetrics):
    def __init__(self):
        super().__init__()
        self.diagnostic_data = defaultdict(dict)

    def comprehensive_diagnostic_evaluation(self, results, relevant_docs, k_values, query):
        """综合诊断评估 - 提供详细的问题分析"""
        evaluation = self.comprehensive_evaluation(results, relevant_docs, k_values)

        # 添加诊断信息
        evaluation['diagnostics'] = self._generate_diagnostics(results, relevant_docs, query)

        return evaluation

    def _generate_diagnostics(self, results, relevant_docs, query):
        """生成诊断信息"""
        diagnostics = {
            'query': query,
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(results),
            'relevant_retrieved': [],
            'missing_relevant': [],
            'ranking_issues': [],
            'score_distribution': []
        }

        # 分析检索到的相关文档
        retrieved_doc_ids = [r['doc_id'] for r in results]

        for doc_id in relevant_docs:
            if doc_id in retrieved_doc_ids:
                rank = retrieved_doc_ids.index(doc_id) + 1
                score = results[rank - 1]['score'] if rank <= len(results) else 0
                diagnostics['relevant_retrieved'].append({
                    'doc_id': doc_id,
                    'rank': rank,
                    'score': score
                })
            else:
                diagnostics['missing_relevant'].append(doc_id)

        # 分析排名问题
        if diagnostics['relevant_retrieved']:
            relevant_scores = [item['score'] for item in diagnostics['relevant_retrieved']]
            non_relevant_scores = [r['score'] for r in results if r['doc_id'] not in relevant_docs]

            if non_relevant_scores and relevant_scores:
                avg_relevant_score = np.mean(relevant_scores)
                avg_non_relevant_score = np.mean(non_relevant_scores)

                if avg_non_relevant_score > avg_relevant_score:
                    diagnostics['ranking_issues'].append(
                        f"非相关文档平均分数({avg_non_relevant_score:.3f})高于相关文档({avg_relevant_score:.3f})"
                    )

        # 分数分布
        if results:
            scores = [r['score'] for r in results]
            diagnostics['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }

        return diagnostics

    def evaluate_retriever_diagnostic(self, retriever, queries, relevance_data, top_k=5):
        """诊断评估检索器"""
        results_summary = {}

        for query in queries:
            try:
                # 执行检索
                search_results = retriever.retrieve(query, top_k=10)  # 获取更多结果用于分析
                relevant_docs = relevance_data.get(query, set())

                # 详细评估
                evaluation = self.comprehensive_diagnostic_evaluation(
                    search_results, relevant_docs, [1, 3, 5, 10], query
                )

                results_summary[query] = evaluation

                # 打印诊断信息
                self._print_query_diagnosis(query, evaluation)

            except Exception as e:
                print(f"查询 '{query}' 评估失败: {e}")
                continue

        return results_summary

    def _print_query_diagnosis(self, query, evaluation):
        """打印查询诊断信息"""
        diagnostics = evaluation.get('diagnostics', {})

        print(f"\n诊断分析 - 查询: '{query}'")
        print(f"相关文档数: {diagnostics.get('num_relevant', 0)}")
        print(f"检索结果数: {diagnostics.get('num_retrieved', 0)}")
        print(f"检索到的相关文档: {len(diagnostics.get('relevant_retrieved', []))}")
        print(f"缺失的相关文档: {len(diagnostics.get('missing_relevant', []))}")

        if diagnostics.get('ranking_issues'):
            print("排名问题:")
            for issue in diagnostics['ranking_issues']:
                print(f"  - {issue}")

        if diagnostics.get('relevant_retrieved'):
            print("检索到的相关文档详情:")
            for doc_info in diagnostics['relevant_retrieved']:
                print(f"  - {doc_info['doc_id']}: 排名#{doc_info['rank']}, 分数:{doc_info['score']:.3f}")