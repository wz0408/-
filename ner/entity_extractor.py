import re
import jieba
import jieba.posseg as pseg
from typing import Dict, List, Any
import os


class EntityExtractor:
    def __init__(self, use_extended_dict=True):
        # 初始化jieba，添加专业词汇
        self._init_jieba()

        # 加载扩展词典
        if use_extended_dict:
            self._load_extended_dict()

        # 各种实体模式
        self._init_patterns()

        print(f"✓ 实体识别器初始化完成，包含 {len(self.university_list)} 所大学，"
              f"{len(self.city_list)} 个城市，{len(self.major_list)} 个专业")

    def _init_jieba(self):
        """初始化jieba分词器，添加专业词汇"""
        # 教育领域词汇
        education_words = [
            '清华大学', '北京大学', '浙江大学', '上海交通大学', '复旦大学',
            '中国人民大学', '北京航空航天大学', '北京理工大学', '北京师范大学',
            '同济大学', '华东师范大学', '上海大学', '上海财经大学', '东华大学',
            '上海外国语大学', '南京大学', '东南大学', '中国科学技术大学',
            '武汉大学', '华中科技大学', '中山大学', '华南理工大学',
            '四川大学', '电子科技大学', '西安交通大学', '哈尔滨工业大学',
            '南开大学', '天津大学', '厦门大学', '山东大学', '吉林大学',
            '大连理工大学', '西北工业大学', '国防科技大学', '中南大学',
            '重庆大学', '兰州大学', '东北大学', '湖南大学', '中国海洋大学',
            '中央民族大学', '西北农林科技大学'
        ]

        # 专业词汇
        major_words = [
            '计算机科学', '软件工程', '人工智能', '数据科学', '机器学习',
            '深度学习', '自然语言处理', '计算机视觉', '大数据', '云计算',
            '物联网', '区块链', '网络安全', '信息安全', '电子信息工程',
            '通信工程', '自动化', '电气工程', '机械工程', '土木工程',
            '建筑学', '材料科学', '化学工程', '生物工程', '临床医学',
            '口腔医学', '药学', '护理学', '经济学', '金融学', '会计学',
            '工商管理', '市场营销', '人力资源管理', '法学', '社会学',
            '心理学', '教育学', '英语语言文学', '日语', '法语', '德语'
        ]

        # 公司词汇
        company_words = [
            '阿里巴巴', '腾讯', '百度', '华为', '小米', '字节跳动',
            '京东', '拼多多', '美团', '滴滴', '网易', '搜狐',
            '新浪', '联想', '中兴', '大疆', '蔚来', '理想',
            '小鹏', '比亚迪', '格力', '海尔', '美的', '华为公司'
        ]

        # 添加所有词汇
        for word in education_words + major_words + company_words:
            jieba.add_word(word)

    def _load_extended_dict(self):
        """加载扩展词典文件"""
        dict_files = [
            "ner/education_dict.txt",
            "ner/tech_dict.txt",
            "ner/location_dict.txt"
        ]

        for dict_file in dict_files:
            if os.path.exists(dict_file):
                try:
                    with open(dict_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word and len(word) > 1:
                                jieba.add_word(word)
                    print(f"✓ 加载词典: {dict_file}")
                except Exception as e:
                    print(f"⚠ 加载词典失败 {dict_file}: {e}")

    def _init_patterns(self):
        """初始化所有实体列表和正则表达式模式"""
        # 大学列表
        self.university_list = self._get_university_list()
        self.university_patterns = [r'\b' + re.escape(uni) + r'\b' for uni in self.university_list]

        # 城市列表
        self.city_list = self._get_city_list()
        self.city_patterns = [r'\b' + re.escape(city) + r'\b' for city in self.city_list]

        # 专业列表
        self.major_list = self._get_major_list()
        self.major_patterns = [r'\b' + re.escape(major) + r'\b' for major in self.major_list]

        # 公司列表
        self.company_list = self._get_company_list()
        self.company_patterns = [r'\b' + re.escape(company) + r'\b' for company in self.company_list]

        # 时间模式
        self.time_patterns = [
            r'\b20[0-9]{2}\b',  # 2000-2099
            r'\b20[0-9]{2}年\b',
            r'\b[0-9]{1,2}月\b',
            r'\b[0-9]{1,2}日\b',
            r'\b今天\b', r'\b明天\b', r'\b昨天\b',
            r'\b上午\b', r'\b下午\b', r'\b晚上\b'
        ]

    def _get_university_list(self):
        """获取大学列表"""
        return [
            # 北京
            '清华大学', '北京大学', '中国人民大学', '北京航空航天大学',
            '北京理工大学', '北京师范大学', '中国农业大学', '北京科技大学',
            '北京交通大学', '北京邮电大学', '北京化工大学', '北京工业大学',
            '北京林业大学', '北京中医药大学', '北京外国语大学',
            '中国传媒大学', '中央财经大学', '对外经济贸易大学',
            '北京体育大学', '中央音乐学院', '中央美术学院', '中央戏剧学院',

            # 上海
            '上海交通大学', '复旦大学', '同济大学', '华东师范大学',
            '上海大学', '上海财经大学', '东华大学', '上海外国语大学',
            '华东理工大学', '上海师范大学', '上海理工大学',

            # 其他地区
            '浙江大学', '南京大学', '中国科学技术大学', '哈尔滨工业大学',
            '西安交通大学', '武汉大学', '华中科技大学', '中山大学',
            '四川大学', '电子科技大学', '南开大学', '天津大学',
            '山东大学', '吉林大学', '大连理工大学', '西北工业大学',
            '中南大学', '重庆大学', '兰州大学', '东北大学',
            '湖南大学', '中国海洋大学', '西北农林科技大学'
        ]

    def _get_city_list(self):
        """获取城市列表"""
        return [
            # 直辖市
            '北京', '上海', '天津', '重庆',

            # 省会城市
            '广州', '深圳', '杭州', '南京', '武汉', '成都', '西安',
            '郑州', '长沙', '合肥', '福州', '厦门', '南昌', '济南',
            '青岛', '石家庄', '太原', '呼和浩特', '沈阳', '大连',
            '长春', '哈尔滨', '苏州', '无锡', '宁波', '温州',
            '南宁', '海口', '贵阳', '昆明', '拉萨', '兰州',
            '西宁', '银川', '乌鲁木齐', '香港', '澳门', '台北'
        ]

    def _get_major_list(self):
        """获取专业列表"""
        return [
            # 计算机相关
            '计算机科学', '计算机', '软件工程', '人工智能', 'AI',
            '机器学习', '深度学习', '自然语言处理', '计算机视觉',
            '数据科学', '大数据', '数据分析', '数据挖掘',
            '云计算', '物联网', '区块链', '网络安全', '信息安全',

            # 工程类
            '电子信息工程', '通信工程', '自动化', '电气工程',
            '机械工程', '土木工程', '建筑学', '城乡规划',
            '材料科学', '化学工程', '环境工程', '生物工程',

            # 经管类
            '经济学', '金融学', '会计学', '财务管理', '工商管理',
            '市场营销', '人力资源管理', '国际贸易', '物流管理'
        ]

    def _get_company_list(self):
        """获取公司列表"""
        return [
            '阿里巴巴', '腾讯', '百度', '华为', '小米', '字节跳动',
            '京东', '拼多多', '美团', '滴滴', '网易', '搜狐',
            '新浪', '联想', '中兴', '大疆', '蔚来', '理想',
            '小鹏', '比亚迪', '格力', '海尔', '美的', '华为公司'
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体 - 改进版本"""
        entities = {}

        # 使用精确匹配（优先）
        entities.update(self._extract_with_exact_match(text))

        # 使用jieba词性标注（辅助）
        entities.update(self._extract_with_jieba(text))

        # 去重并过滤空结果
        for entity_type, entity_list in list(entities.items()):
            if entity_list:
                entities[entity_type] = list(set(entity_list))
            else:
                del entities[entity_type]

        return entities

    def _extract_with_exact_match(self, text: str) -> Dict[str, List[str]]:
        """使用精确匹配提取实体"""
        entities = {}

        # 提取大学
        universities = []
        for pattern in self.university_patterns:
            matches = re.findall(pattern, text)
            universities.extend(matches)
        if universities:
            entities['UNIVERSITY'] = universities

        # 提取城市
        cities = []
        for pattern in self.city_patterns:
            matches = re.findall(pattern, text)
            cities.extend(matches)
        if cities:
            entities['CITY'] = cities

        # 提取专业
        majors = []
        for pattern in self.major_patterns:
            matches = re.findall(pattern, text)
            majors.extend(matches)
        if majors:
            entities['MAJOR'] = majors

        # 提取公司
        companies = []
        for pattern in self.company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        if companies:
            entities['COMPANY'] = companies

        # 提取时间
        times = []
        for pattern in self.time_patterns:
            times.extend(re.findall(pattern, text))
        if times:
            entities['TIME'] = times

        return entities

    def _extract_with_jieba(self, text: str) -> Dict[str, List[str]]:
        """使用jieba词性标注提取实体 - 改进版本"""
        entities = {}
        words = pseg.cut(text)

        university_set = set(self.university_list)
        city_set = set(self.city_list)
        major_set = set(self.major_list)
        company_set = set(self.company_list)

        for word, flag in words:
            # 只匹配我们预定义列表中的实体，避免错误识别
            if word in university_set:
                if 'UNIVERSITY' not in entities:
                    entities['UNIVERSITY'] = []
                entities['UNIVERSITY'].append(word)

            elif word in city_set:
                if 'CITY' not in entities:
                    entities['CITY'] = []
                entities['CITY'].append(word)

            elif word in major_set:
                if 'MAJOR' not in entities:
                    entities['MAJOR'] = []
                entities['MAJOR'].append(word)

            elif word in company_set:
                if 'COMPANY' not in entities:
                    entities['COMPANY'] = []
                entities['COMPANY'].append(word)

            # 时间识别（相对宽松）
            elif flag == 't' or '年' in word or '月' in word or '日' in word:
                if 'TIME' not in entities:
                    entities['TIME'] = []
                # 过滤掉太短的时间词
                if len(word) > 1:
                    entities['TIME'].append(word)

        return entities

    def extract_from_results(self, search_results: List[Dict]) -> List[Dict]:
        """从搜索结果中提取实体"""
        enriched_results = []

        for result in search_results:
            result_copy = result.copy()
            result_copy['entities'] = self.extract_entities(result['text'])
            enriched_results.append(result_copy)

        return enriched_results