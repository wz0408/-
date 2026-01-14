from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_retriever import BaseRetriever
import numpy as np
import jieba


class TfidfRetriever(BaseRetriever):
    def __init__(self, max_features=5000, min_score=0.1):
        super().__init__()
        self.max_features = max_features
        self.min_score = min_score
        self.vectorizer, self.tfidf_matrix = self._build_tfidf_index()

        print(f"✓ TF-IDF检索器初始化完成，文档数: {len(self.documents)}")

    def _build_tfidf_index(self):
        """构建TF-IDF索引 - 改进版本"""
        # 准备语料库
        corpus = []
        for doc_content in self.documents.values():
            if not doc_content or len(doc_content.strip()) == 0:
                # 处理空文档
                corpus.append("empty document")
            else:
                # 使用改进的分词
                tokens = self._advanced_tokenize(doc_content)
                corpus.append(' '.join(tokens))

        # 如果所有文档都为空，创建默认语料
        if not corpus or all(doc == "empty document" for doc in corpus):
            corpus = ["default document"] * len(self.documents)

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            tokenizer=lambda x: x.split(),
            token_pattern=None,
            min_df=1,  # 至少出现在1个文档中
            max_df=0.95  # 最多出现在95%的文档中
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            return vectorizer, tfidf_matrix
        except Exception as e:
            print(f"TF-IDF索引构建失败: {e}")
            # 返回一个基础的向量器
            return self._create_basic_vectorizer(corpus)

    def _create_basic_vectorizer(self, corpus):
        """创建基础向量器作为备选"""
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return vectorizer, tfidf_matrix

    def _advanced_tokenize(self, text):
        """改进的分词函数"""
        words = jieba.cut(text.lower())
        tokens = []

        for word in words:
            word = word.strip()
            if len(word) > 0:
                # 过滤停用词和单个字符（除了数字和字母）
                if len(word) > 1 or word.isalnum():
                    tokens.append(word)

        return tokens if tokens else ['document']

    def retrieve(self, query, top_k=5):
        """检索 - 使用增强的查询扩展"""
        if not query or not query.strip():
            return []

        # 深度查询扩展
        expanded_query = self.expand_query_with_intent(query)



        query = self.preprocess_query(query)
        query_tokens = self._advanced_tokenize(query)
        query_processed = ' '.join(query_tokens)

        if not query_processed.strip():
            return self._get_fallback_results(query, top_k)

        try:
            # 转换查询向量
            query_vec = self.vectorizer.transform([query_processed])

            # 计算相似度
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            # 处理异常值
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
            similarities = np.clip(similarities, 0, 1)

            # 如果所有分数都为0，给一个基础分数
            if np.max(similarities) == 0:
                similarities = np.array([self.min_score] * len(similarities))
            else:
                # 确保最小分数
                similarities = np.maximum(similarities, self.min_score)

            # 排序并返回结果
            doc_scores = list(zip(self.doc_ids, similarities))
            ranked_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)

            results = []
            for i, (doc_id, score) in enumerate(ranked_results[:top_k]):
                if doc_id in self.documents:
                    results.append({
                        'doc_id': doc_id,
                        'text': self.documents[doc_id],
                        'score': float(score),
                        'retriever': 'tfidf',
                        'rank': i + 1
                    })

            # 补充结果
            if len(results) < top_k:
                results.extend(self._get_additional_results(results, top_k - len(results)))

            return results

        except Exception as e:
            print(f"TF-IDF检索错误: {e}")
            return self._get_fallback_results(query, top_k)

    def _get_fallback_results(self, query, top_k):
        """备选检索方案"""
        results = []
        for i, doc_id in enumerate(self.doc_ids[:top_k]):
            score = self._simple_text_match(query, self.documents[doc_id])
            results.append({
                'doc_id': doc_id,
                'text': self.documents[doc_id],
                'score': score,
                'retriever': 'tfidf_fallback',
                'rank': i + 1
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _get_additional_results(self, existing_results, count):
        """获取额外的结果"""
        additional = []
        existing_doc_ids = set(r['doc_id'] for r in existing_results)

        for doc_id in self.doc_ids:
            if doc_id not in existing_doc_ids and len(additional) < count:
                additional.append({
                    'doc_id': doc_id,
                    'text': self.documents[doc_id],
                    'score': self.min_score / 2,
                    'retriever': 'tfidf',
                    'rank': len(existing_results) + len(additional) + 1
                })

        return additional

    def _simple_text_match(self, query, document):
        """简单的文本匹配分数"""
        if not query or not document:
            return self.min_score

        query_terms = set(self._advanced_tokenize(query))
        doc_terms = set(self._advanced_tokenize(document))

        if not query_terms:
            return self.min_score

        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)

        if union == 0:
            return self.min_score

        similarity = intersection / union
        return max(similarity, self.min_score)