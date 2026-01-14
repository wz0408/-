import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 模型配置
DEFAULT_TOP_K = 3

# 评估配置
METRICS = ['precision@k', 'recall@k', 'f1@k', 'mrr', 'map', 'ndcg@k']
K_VALUES = [1, 3, 5, 10]

# 检索器配置
RETRIEVAL_METHODS = ['bm25', 'tfidf', 'neural', 'hybrid']