import time
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from retrieval.bm25_retriever import BM25Retriever
from retrieval.tfidf_retriever import TfidfRetriever
from retrieval.hybrid_retriever import HybridRetriever
from ner.entity_extractor import EntityExtractor
from evaluation.metrics import EvaluationMetrics
from config.relevance_data import RELEVANCE_DATA, TEST_QUERIES, QUERY_CATEGORIES


class AdvancedSearchSystem:
    def __init__(self, use_hybrid=True):
        self.use_hybrid = use_hybrid
        self.entity_extractor = EntityExtractor(use_extended_dict=True)
        self.evaluator = EvaluationMetrics()

        # 初始化检索器
        self._initialize_retrievers()

        print("=" * 70)
        print("高级信息检索系统 (v2.0)")
        print("=" * 70)
        print("系统初始化完成！")
        print(f"可用检索器: {list(self.retrievers.keys())}")
        print(f"测试查询数量: {len(TEST_QUERIES)}")
        print(f"实体类型: 大学、城市、专业、公司、人物、时间等")

    def _initialize_retrievers(self):
        """初始化各种检索器"""
        self.retrievers = {}

        print("正在初始化检索器...")
        start_time = time.time()

        # 基础检索器
        try:
            self.retrievers['bm25'] = BM25Retriever()
            print("✓ BM25检索器初始化完成")
        except Exception as e:
            print(f"✗ BM25检索器初始化失败: {e}")

        try:
            self.retrievers['tfidf'] = TfidfRetriever(max_features=10000)
            print("✓ TF-IDF检索器初始化完成")
        except Exception as e:
            print(f"✗ TF-IDF检索器初始化失败: {e}")

        # 混合检索器
        if self.use_hybrid:
            try:
                self.retrievers['hybrid'] = HybridRetriever(methods=['bm25', 'tfidf'])
                print("✓ 混合检索器初始化完成")
            except Exception as e:
                print(f"✗ 混合检索器初始化失败: {e}")

        init_time = time.time() - start_time
        print(f"检索器初始化总耗时: {init_time:.2f}秒")

    def run_comprehensive_evaluation(self, output_file="evaluation_report.json"):
        """运行综合评估"""
        print("\n" + "=" * 70)
        print("综合评估报告生成中...")
        print("=" * 70)

        all_results = {}
        detailed_metrics = {}

        for retriever_name in self.retrievers.keys():
            print(f"\n评估检索器: {retriever_name}")
            print("-" * 50)

            retriever_results = {}
            query_metrics = {}

            for query in TEST_QUERIES:
                try:
                    # 执行搜索
                    search_result = self.search(query, retriever_name, top_k=10, extract_entities=False)

                    if search_result and search_result['results']:
                        # 获取相关文档
                        relevant_docs = RELEVANCE_DATA.get(query, set())

                        # 评估
                        metrics = self.evaluator.comprehensive_evaluation(
                            search_result['results'],
                            relevant_docs,
                            [1, 3, 5, 10]
                        )

                        query_metrics[query] = {
                            'metrics': metrics,
                            'num_results': len(search_result['results']),
                            'retrieval_time': search_result['stats']['retrieval_time']
                        }

                        retriever_results[query] = search_result['results']

                        # 显示进度
                        print(f"  {query}: P@3={metrics['precision@3']:.3f}, "
                              f"R@3={metrics['recall@3']:.3f}, "
                              f"F1@3={metrics['f1@3']:.3f}")

                except Exception as e:
                    print(f"  查询 '{query}' 评估失败: {e}")
                    continue

            all_results[retriever_name] = retriever_results
            detailed_metrics[retriever_name] = query_metrics

        # 生成汇总报告
        summary = self._generate_evaluation_summary(detailed_metrics)

        # 保存结果
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {
                'total_queries': len(TEST_QUERIES),
                'queries_by_category': QUERY_CATEGORIES
            },
            'detailed_metrics': detailed_metrics,
            'summary': summary
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 评估报告已保存至: {output_file}")

        # 显示汇总结果
        self._display_evaluation_summary(summary)

        return report

    def _generate_evaluation_summary(self, detailed_metrics):
        """生成评估汇总"""
        summary = {}

        for retriever_name, query_metrics in detailed_metrics.items():
            if not query_metrics:
                continue

            # 计算各类指标的平均值
            metrics_to_avg = ['precision@1', 'precision@3', 'precision@5',
                              'recall@1', 'recall@3', 'recall@5',
                              'f1@1', 'f1@3', 'f1@5',
                              'ndcg@1', 'ndcg@3', 'ndcg@5', 'ap']

            avg_metrics = {}
            for metric in metrics_to_avg:
                values = [qm['metrics'].get(metric, 0) for qm in query_metrics.values()]
                avg_metrics[metric] = np.mean(values) if values else 0

            # 计算各查询类别的平均性能
            category_performance = {}
            for category, queries in QUERY_CATEGORIES.items():
                cat_metrics = []
                for query in queries:
                    if query in query_metrics:
                        cat_metrics.append(query_metrics[query]['metrics'].get('f1@3', 0))
                if cat_metrics:
                    category_performance[category] = np.mean(cat_metrics)

            # 计算平均检索时间
            avg_time = np.mean([qm['retrieval_time'] for qm in query_metrics.values()])

            summary[retriever_name] = {
                'avg_metrics': avg_metrics,
                'category_performance': category_performance,
                'avg_retrieval_time': avg_time,
                'num_queries_evaluated': len(query_metrics)
            }

        return summary

    def _display_evaluation_summary(self, summary):
        """显示评估汇总"""
        print("\n" + "=" * 70)
        print("评估结果汇总")
        print("=" * 70)

        # 表头
        print(f"{'检索器':<15} {'MAP':<8} {'P@3':<8} {'R@3':<8} {'F1@3':<8} {'NDCG@3':<8} {'时间(ms)':<10}")
        print("-" * 70)

        for retriever_name, retriever_summary in summary.items():
            avg_metrics = retriever_summary['avg_metrics']
            map_score = avg_metrics.get('ap', 0)
            p3 = avg_metrics.get('precision@3', 0)
            r3 = avg_metrics.get('recall@3', 0)
            f3 = avg_metrics.get('f1@3', 0)
            ndcg3 = avg_metrics.get('ndcg@3', 0)
            avg_time = retriever_summary['avg_retrieval_time'] * 1000  # 转换为毫秒

            print(f"{retriever_name:<15} {map_score:<8.4f} {p3:<8.4f} {r3:<8.4f} "
                  f"{f3:<8.4f} {ndcg3:<8.4f} {avg_time:<10.2f}")

        # 显示各查询类别的性能
        print("\n" + "=" * 70)
        print("各查询类别F1@3分数")
        print("=" * 70)

        categories = list(QUERY_CATEGORIES.keys())
        print(f"{'检索器':<15}", end="")
        for category in categories:
            print(f" {category:<10}", end="")
        print()
        print("-" * 70)

        for retriever_name, retriever_summary in summary.items():
            print(f"{retriever_name:<15}", end="")
            for category in categories:
                score = retriever_summary['category_performance'].get(category, 0)
                print(f" {score:<10.4f}", end="")
            print()

    def analyze_entity_coverage(self, num_queries=5):
        """分析实体覆盖情况"""
        print("\n" + "=" * 70)
        print("实体覆盖分析")
        print("=" * 70)

        sample_queries = TEST_QUERIES[:min(num_queries, len(TEST_QUERIES))]

        for query in sample_queries:
            print(f"\n查询: '{query}'")
            result = self.search(query, 'hybrid', top_k=3, extract_entities=True)

            if result and result['results']:
                print(f"返回 {len(result['results'])} 个结果:")

                entity_stats = defaultdict(set)
                for i, r in enumerate(result['results']):
                    print(f"  {i + 1}. {r['doc_id']} (分数: {r['score']:.4f})")

                    if 'entities' in r and r['entities']:
                        for entity_type, entities in r['entities'].items():
                            for entity in entities:
                                entity_stats[entity_type].add(entity)

                # 统计实体类型
                if entity_stats:
                    print("  识别的实体类型:")
                    for entity_type, entities in entity_stats.items():
                        print(f"    {entity_type}: {list(entities)[:5]}")  # 只显示前5个
                        if len(entities) > 5:
                            print(f"      ... 共 {len(entities)} 个")
                else:
                    print("  未识别到实体")
            else:
                print("  无结果")

    def run_performance_benchmark(self, num_iterations=10):
        """运行性能基准测试"""
        print("\n" + "=" * 70)
        print("性能基准测试")
        print("=" * 70)

        test_queries = ["北京 大学", "计算机 科学", "人工智能", "上海 经济"]

        for retriever_name in self.retrievers.keys():
            print(f"\n测试检索器: {retriever_name}")

            times = []
            for query in test_queries:
                for _ in range(num_iterations):
                    start_time = time.time()
                    self.search(query, retriever_name, top_k=5, extract_entities=False)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # 转换为毫秒

            avg_time = np.mean(times)
            std_time = np.std(times)

            print(f"  平均检索时间: {avg_time:.2f} ms")
            print(f"  标准差: {std_time:.2f} ms")
            print(f"  最大时间: {max(times):.2f} ms")
            print(f"  最小时间: {min(times):.2f} ms")

    def search(self, query: str, retriever_type: str = None, top_k: int = 5, extract_entities: bool = True):
        """执行搜索"""
        if not retriever_type:
            retriever_type = 'hybrid' if 'hybrid' in self.retrievers else list(self.retrievers.keys())[0]

        if retriever_type not in self.retrievers:
            available = list(self.retrievers.keys())
            print(f"可用检索器: {available}")
            raise ValueError(f"不支持的检索器类型: {retriever_type}")

        start_time = time.time()

        try:
            # 执行检索
            results = self.retrievers[retriever_type].retrieve(query, top_k)
            retrieval_time = time.time() - start_time

            # 实体提取
            if extract_entities and results:
                entity_start = time.time()
                for result in results:
                    result['entities'] = self.entity_extractor.extract_entities(result['text'])
                entity_time = time.time() - entity_start
            else:
                entity_time = 0

            return {
                'query': query,
                'retriever': retriever_type,
                'results': results,
                'stats': {
                    'retrieval_time': retrieval_time,
                    'entity_time': entity_time,
                    'total_time': retrieval_time + entity_time,
                    'num_results': len(results)
                }
            }
        except Exception as e:
            print(f"搜索过程中发生错误: {e}")
            return None

    def interactive_search_loop(self):
        """交互式搜索循环"""
        print("\n" + "=" * 70)
        print("交互式搜索模式")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'compare' 比较检索器性能")
        print("输入 'analyze' 分析实体覆盖")
        print("输入 'benchmark' 运行性能测试")
        print("输入 'help' 查看帮助")
        print("=" * 70)

        while True:
            print("\n" + "-" * 40)
            command = input("请输入命令或搜索查询: ").strip()

            if command.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break

            elif command.lower() == 'compare':
                self.run_comprehensive_evaluation()
                continue

            elif command.lower() == 'analyze':
                self.analyze_entity_coverage()
                continue

            elif command.lower() == 'benchmark':
                self.run_performance_benchmark()
                continue

            elif command.lower() == 'help':
                self._show_help()
                continue

            elif command.lower() == '':
                continue

            # 执行搜索
            print(f"\n正在搜索: '{command}'")

            # 让用户选择检索器
            print("\n可用检索器:")
            retrievers = list(self.retrievers.keys())
            for i, retriever in enumerate(retrievers, 1):
                print(f"{i}. {retriever}")

            choice = input("选择检索器 (回车使用默认混合检索器): ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(retrievers):
                retriever_type = retrievers[int(choice) - 1]
            else:
                retriever_type = 'hybrid' if 'hybrid' in retrievers else retrievers[0]
                print(f"使用默认检索器: {retriever_type}")

            # 执行搜索
            result = self.search(command, retriever_type, top_k=5)
            self._display_search_results(result)

    def _show_help(self):
        """显示帮助信息"""
        print("\n帮助信息:")
        print("  1. 直接输入查询语句进行搜索")
        print("  2. 命令:")
        print("     - quit/exit/q: 退出系统")
        print("     - compare: 比较所有检索器的性能")
        print("     - analyze: 分析实体覆盖情况")
        print("     - benchmark: 运行性能基准测试")
        print("     - help: 显示此帮助信息")
        print("  3. 搜索示例:")
        print("     - 北京 大学")
        print("     - 计算机 科学")
        print("     - 人工智能")
        print("     - 上海 经济")
        print("     - Python 编程")

    def _display_search_results(self, search_result):
        """显示搜索结果"""
        if not search_result or not search_result['results']:
            print("未找到相关结果")
            return

        print(f"\n查询: '{search_result['query']}'")
        print(f"检索器: {search_result['retriever']}")
        print(f"找到 {len(search_result['results'])} 个结果 (耗时: {search_result['stats']['total_time']:.3f}s)")
        print("=" * 80)

        for i, result in enumerate(search_result['results'], 1):
            print(f"\n{i}. [{result['doc_id']}] 分数: {result['score']:.4f}")
            print(f"   {result['text'][:100]}...")

            if 'entities' in result and result['entities']:
                print("   实体: ", end="")
                entity_strs = []
                for entity_type, entities in result['entities'].items():
                    if entities:
                        entity_strs.append(f"{entity_type}: {entities}")
                print("; ".join(entity_strs))

        print("=" * 80)


def main():
    """主函数"""
    try:
        # 初始化系统
        system = AdvancedSearchSystem(use_hybrid=True)

        # 运行综合评估
        system.run_comprehensive_evaluation()

        # 分析实体覆盖
        system.analyze_entity_coverage()

        # 运行性能基准测试
        system.run_performance_benchmark()

        # 进入交互模式
        system.interactive_search_loop()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n系统发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()