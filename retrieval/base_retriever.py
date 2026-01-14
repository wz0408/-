from abc import ABC, abstractmethod
from utils.data_loader import DataLoader


class BaseRetriever(ABC):
    def __init__(self):
        self.data_loader = DataLoader()
        self.doc_ids = self.data_loader.get_document_ids()
        self.documents = dict(self.data_loader.get_all_documents())

        # 同义词扩展词典
        self.synonyms = {
            '计算机': ['计算机', '电脑', '计算机科学', '计算'],
            '人工智能': ['人工智能', 'AI', '智能', '机器学习'],
            '大数据': ['大数据', '数据', '数据分析', '数据科学'],
            '北京': ['北京', '京城', '帝都'],
            '上海': ['上海', '沪', '申城'],
            '大学': ['大学', '高校', '高等院校']
        }

    @abstractmethod
    def retrieve(self, query, top_k=3):
        """检索抽象方法"""
        pass

    def preprocess_query(self, query):
        """查询预处理"""
        return query.lower().strip()

    def expand_query(self, query):
        """查询扩展 - 添加同义词"""
        original_terms = self._advanced_tokenize(query)
        expanded_terms = set(original_terms)

        for term in original_terms:
            if term in self.synonyms:
                expanded_terms.update(self.synonyms[term])

        # 返回扩展后的查询字符串
        return ' '.join(expanded_terms)

    def _advanced_tokenize(self, text):
        """改进的分词函数"""
        import jieba
        words = jieba.cut_for_search(text.lower())

        tokens = []
        for word in words:
            word = word.strip()
            if len(word) > 0 and (len(word) > 1 or word.isalnum()):
                tokens.append(word)

        return tokens if tokens else ['document']

    # 在base_retriever.py中增强查询理解
    def understand_query(self, query):
        """深度理解查询意图"""
        # 检测查询类型
        query_lower = query.lower()

        intent = {
            'type': 'general',
            'entities': [],
            'concepts': []
        }

        # 教育相关查询
        education_terms = ['大学', '学院', '专业', '招生', '教育', '学习']
        if any(term in query_lower for term in education_terms):
            intent['type'] = 'education'
            intent['concepts'].extend(['教育', '学术'])

        # 科技相关查询
        tech_terms = ['计算机', '编程', '人工智能', 'AI', '数据', '软件']
        if any(term in query_lower for term in tech_terms):
            intent['type'] = 'technology'
            intent['concepts'].extend(['技术', '创新'])

        # 经济相关查询
        economy_terms = ['经济', '金融', '市场', '投资', '商业']
        if any(term in query_lower for term in economy_terms):
            intent['type'] = 'economy'
            intent['concepts'].extend(['经济', '商业'])

        # 提取具体实体
        import jieba.posseg as pseg
        words = pseg.cut(query)
        for word, flag in words:
            if flag == 'ns':  # 地名
                intent['entities'].append(('location', word))
            elif '大学' in word or '学院' in word:
                intent['entities'].append(('university', word))
            elif flag == 'n' and len(word) > 1:  # 名词
                intent['concepts'].append(word)

        return intent

    def expand_query_with_intent(self, query):
        """基于查询意图进行扩展"""
        intent = self.understand_query(query)
        expanded_terms = set(self._advanced_tokenize(query))

        # 基于意图添加相关术语
        if intent['type'] == 'education':
            expanded_terms.update(['教育', '学术', '大学', '学习'])
        elif intent['type'] == 'technology':
            expanded_terms.update(['技术', '创新', '科学', '工程'])
        elif intent['type'] == 'economy':
            expanded_terms.update(['经济', '商业', '市场', '金融'])

        # 添加实体相关术语
        for entity_type, entity in intent['entities']:
            if entity_type == 'university':
                expanded_terms.update([entity, '高校', '高等教育'])
            elif entity_type == 'location':
                expanded_terms.update([entity, '城市', '地区'])

        return ' '.join(expanded_terms)