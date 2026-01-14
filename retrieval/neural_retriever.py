import numpy as np
from .base_retriever import BaseRetriever


class NeuralRetriever(BaseRetriever):
    def __init__(self, use_simple_model=True):
        super().__init__()
        self.use_simple_model = use_simple_model

        if not use_simple_model:
            # 使用高级模型
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("成功加载SentenceTransformer模型")
                self.advanced = True
            except ImportError:
                print("SentenceTransformer不可用，将使用简单向量模型")
                self.advanced = False
                self.model = None
        else:
            self.advanced = False
            self.model = None

        # 构建索引
        self.index, self.doc_embeddings = self._build_index()

    def _build_index(self):
        """构建索引 - 使用简单向量模型"""
        doc_texts = list(self.documents.values())

        if self.advanced and self.model is not None:
            # 使用高级模型
            print("正在生成文档嵌入向量...")
            doc_embeddings = self.model.encode(doc_texts, normalize_embeddings=True)
        else:
            # 使用简单词向量模型
            print("使用简单词向量模型...")
            doc_embeddings = self._simple_text_to_vector(doc_texts)

        return None, doc_embeddings

    def _simple_text_to_vector(self, texts):
        """简单的文本到向量转换"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 使用TF-IDF创建文档向量
        vectorizer = TfidfVectorizer(max_features=300)

        # 分词并连接
        corpus = [' '.join(self.data_loader.tokenize(text)) for text in texts]

        vectors = vectorizer.fit_transform(corpus).toarray()

        # 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        vectors = vectors / norms

        return vectors

    def _query_to_vector(self, query):
        """查询文本到向量转换"""
        if self.advanced and self.model is not None:
            return self.model.encode([query], normalize_embeddings=True)
        else:
            # 与文档使用相同的向量化方法
            from sklearn.feature_extraction.text import TfidfVectorizer

            # 获取所有文档用于构建相同的向量空间
            all_texts = list(self.documents.values())
            corpus = [' '.join(self.data_loader.tokenize(text)) for text in all_texts]
            corpus.append(' '.join(self.data_loader.tokenize(query)))

            vectorizer = TfidfVectorizer(max_features=300)
            all_vectors = vectorizer.fit_transform(corpus).toarray()

            # 查询向量是最后一个
            query_vector = all_vectors[-1:].astype('float32')

            # 归一化
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            return query_vector

    def retrieve(self, query, top_k=3):
        """神经检索 - 使用简单向量模型"""
        query = self.preprocess_query(query)

        # 生成查询向量
        query_vector = self._query_to_vector(query)

        # 计算余弦相似度
        similarities = np.dot(self.doc_embeddings, query_vector.T).flatten()

        # 排序
        sorted_indices = np.argsort(-similarities)  # 降序

        results = []
        for i, idx in enumerate(sorted_indices[:top_k]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                results.append({
                    'doc_id': doc_id,
                    'text': self.documents[doc_id],
                    'score': float(similarities[idx]),
                    'retriever': 'neural_simple',
                    'rank': i + 1
                })

        return results