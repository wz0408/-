"""
高级信息检索系统 - 完整测试脚本
测试扩展后的系统功能
"""

import time
# 修复导入 - 从 config.relevance_data 导入
from config.relevance_data import TEST_QUERIES, RELEVANCE_DATA
from retrieval.hybrid_retriever import HybridRetriever
from ner.entity_extractor import EntityExtractor
from evaluation.metrics import EvaluationMetrics

print("高级信息检索系统 - 完整测试")
print("=" * 70)

# 记录开始时间
total_start = time.time()

# 1. 初始化系统
print("\n1. 初始化系统...")
start = time.time()

# 创建检索器
retrievers = {}
try:
    from retrieval.bm25_retriever import BM25Retriever
    retrievers['bm25'] = BM25Retriever()
    print("✓ BM25检索器初始化完成")
except Exception as e:
    print(f"✗ BM25检索器初始化失败: {e}")

try:
    from retrieval.tfidf_retriever import TfidfRetriever
    retrievers['tfidf'] = TfidfRetriever(max_features=10000)
    print("✓ TF-IDF检索器初始化完成")
except Exception as e:
    print(f"✗ TF-IDF检索器初始化失败: {e}")

try:
    retrievers['hybrid'] = HybridRetriever(methods=['bm25', 'tfidf'])
    print("✓ 混合检索器初始化完成")
except Exception as e:
    print(f"✗ 混合检索器初始化失败: {e}")

# 创建实体识别器
try:
    entity_extractor = EntityExtractor(use_extended_dict=True)
    print("✓ 实体识别器初始化完成")
except Exception as e:
    print(f"✗ 实体识别器初始化失败: {e}")
    entity_extractor = None

# 创建评估器
evaluator = EvaluationMetrics()
print("✓ 评估器初始化完成")

init_time = time.time() - start
print(f"系统初始化时间: {init_time:.2f}秒")

# 2. 定义搜索函数
def search_system(query, retriever_name='hybrid', top_k=5, extract_entities=True):
    """执行搜索"""
    if retriever_name not in retrievers:
        print(f"检索器 {retriever_name} 不存在")
        return None

    start_time = time.time()

    try:
        # 执行检索
        retriever = retrievers[retriever_name]
        results = retriever.retrieve(query, top_k)
        retrieval_time = time.time() - start_time

        # 实体提取
        if extract_entities and results and entity_extractor:
            entity_start = time.time()
            for result in results:
                result['entities'] = entity_extractor.extract_entities(result['text'])
            entity_time = time.time() - entity_start
        else:
            entity_time = 0

        return {
            'query': query,
            'retriever': retriever_name,
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

# 3. 测试基本搜索
print("\n2. 测试基本搜索功能")
# 使用配置文件中的测试查询
test_queries = TEST_QUERIES[:4]  # 只取前4个查询

for query in test_queries:
    start = time.time()
    result = search_system(query, 'hybrid', top_k=3)
    search_time = time.time() - start

    if result:
        print(f"\n查询: '{query}'")
        print(f"  时间: {search_time:.3f}秒")
        print(f"  结果数: {len(result['results'])}")

        for i, r in enumerate(result['results']):
            print(f"  {i+1}. {r['doc_id']} (分数: {r['score']:.4f})")

            # 检查实体识别
            if 'entities' in r and r['entities']:
                entities_found = []
                for entity_type, entities in r['entities'].items():
                    if entities:
                        entities_found.append(f"{entity_type}:{len(entities)}")
                if entities_found:
                    print(f"     实体: {', '.join(entities_found)}")

# 4. 测试不同检索器
print("\n" + "=" * 70)
print("3. 比较不同检索器")
print("=" * 70)

query = test_queries[0]  # 使用第一个查询
for retriever in ['bm25', 'tfidf', 'hybrid']:
    if retriever in retrievers:
        start = time.time()
        result = search_system(query, retriever, top_k=3, extract_entities=False)
        search_time = time.time() - start

        if result and result['results']:
            scores = [r['score'] for r in result['results']]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"{retriever:<10} 时间: {search_time:.3f}s, 平均分数: {avg_score:.4f}")

# 5. 测试实体识别扩展
print("\n" + "=" * 70)
print("4. 测试扩展的实体识别")
print("=" * 70)

# 测试扩展后的实体识别
test_texts = [
    "上海交通大学是中国著名的大学，计算机科学专业很强。",
    "复旦大学位于上海，人工智能研究方向在全国排名前列。",
    "华为公司在深圳，是中国的科技巨头。",
    "2023年Python编程语言在人工智能领域应用广泛。"
]

if entity_extractor:
    for i, text in enumerate(test_texts, 1):
        entities = entity_extractor.extract_entities(text)
        print(f"\n测试文本 {i}: {text[:50]}...")
        for entity_type, entity_list in entities.items():
            if entity_list:  # 只显示有结果的类型
                print(f"  {entity_type}: {entity_list}")

# 6. 评估功能测试
print("\n" + "=" * 70)
print("5. 评估功能测试")
print("=" * 70)

# 创建测试结果进行评估
test_results = [
    {'doc_id': 'doc1.txt', 'text': '清华大学是北京最好的大学。', 'score': 0.9},
    {'doc_id': 'doc3.txt', 'text': '北京大学计算机科学专业强。', 'score': 0.8},
    {'doc_id': 'doc2.txt', 'text': '北京有多所顶尖大学。', 'score': 0.7},
    {'doc_id': 'doc5.txt', 'text': '上海交通大学在上海。', 'score': 0.6},
    {'doc_id': 'doc7.txt', 'text': '浙江大学在杭州。', 'score': 0.5},
]

relevant_docs = {'doc1.txt', 'doc3.txt', 'doc5.txt'}

evaluation = evaluator.comprehensive_evaluation(test_results, relevant_docs, [1, 3, 5])
print("评估结果摘要:")
for k in [1, 3, 5]:
    print(f"  Precision@{k}: {evaluation[f'precision@{k}']:.4f}")
    print(f"  Recall@{k}: {evaluation[f'recall@{k}']:.4f}")
    print(f"  F1@{k}: {evaluation[f'f1@{k}']:.4f}")
    print(f"  NDCG@{k}: {evaluation[f'ndcg@{k}']:.4f}")

print(f"  MAP: {evaluation['ap']:.4f}")

# 7. 运行综合评估
print("\n" + "=" * 70)
print("6. 运行综合评估")
print("=" * 70)

# 修复这里：使用正确的变量名，没有空格
sample_queries = TEST_QUERIES[:3]  # 只取前3个查询
summary_results = {}

for retriever_name in ['bm25', 'tfidf', 'hybrid']:
    if retriever_name in retrievers:
        print(f"\n评估 {retriever_name}:")
        query_metrics = []

        for query in sample_queries:
            result = search_system(query, retriever_name, top_k=5, extract_entities=False)
            if result and result['results']:
                relevant_docs_set = RELEVANCE_DATA.get(query, set())
                metrics = evaluator.comprehensive_evaluation(
                    result['results'], relevant_docs_set, [3]
                )
                query_metrics.append(metrics)
                print(f"  {query}: P@3={metrics['precision@3']:.3f}, R@3={metrics['recall@3']:.3f}")

        if query_metrics:
            avg_p3 = sum(m['precision@3'] for m in query_metrics) / len(query_metrics)
            avg_r3 = sum(m['recall@3'] for m in query_metrics) / len(query_metrics)
            summary_results[retriever_name] = {'P@3': avg_p3, 'R@3': avg_r3}

# 8. 总结
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)

if summary_results:
    print("\n检索器性能对比:")
    for retriever_name, metrics in summary_results.items():
        print(f"{retriever_name:<10} 平均P@3: {metrics['P@3']:.4f}, 平均R@3: {metrics['R@3']:.4f}")

total_time = time.time() - total_start
print(f"\n✅ 所有测试完成!")
print(f"总测试时间: {total_time:.2f}秒")
print("系统功能完整，可以按照黑板上的结构撰写报告。")

# 9. 生成报告框架提示
print("\n" + "=" * 70)
print("项目报告框架 (对应黑板结构)")
print("=" * 70)
print("""
1. Introduction
   - Problem: 信息检索中的语义匹配与排序问题
   - Existing Method: 单一检索器(BM25/TF-IDF)的局限性
   - Solution: 提出混合检索系统 + 实体识别

2. Related Work
   - 信息检索技术发展
   - 实体识别在检索中的应用
   - 评估指标研究

3. Model
   - 系统架构图(检索 → 实体识别 → 评估)
   - 算法原理(BM25, TF-IDF, 加权融合)
   - 实体识别方法(规则+词典)

4. EXP: Experiment
   - DATA SET: 8个文档 → 扩展后的数据集
   - COMPARISON: BM25 vs TF-IDF vs Hybrid对比
   - ANALYSIS: 结果分析、实体识别效果、系统局限性
""")