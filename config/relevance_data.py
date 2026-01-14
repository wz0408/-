# 相关文档标注数据
# 这里我们为新的数据集定义相关文档
# 文档命名格式: doc1.txt, doc2.txt, ... docN.txt

# 修复相关文档标注
RELEVANCE_DATA = {
    # 教育相关查询
    "北京 大学": {
        "doc1.txt", "doc3.txt", "doc5.txt", "doc7.txt", "doc9.txt", "doc11.txt", "doc17.txt"
    },
    "清华大学": {
        "doc1.txt", "doc3.txt", "doc8.txt", "doc12.txt", "doc15.txt"
    },
    "计算机 科学": {
        "doc1.txt", "doc2.txt", "doc4.txt", "doc6.txt", "doc10.txt", "doc15.txt", "doc18.txt"
    },
    "人工智能": {
        "doc2.txt", "doc4.txt", "doc5.txt", "doc6.txt", "doc11.txt", "doc18.txt", "doc22.txt"
    },
    "上海 大学": {
        "doc3.txt", "doc5.txt", "doc7.txt", "doc12.txt", "doc13.txt", "doc16.txt", "doc20.txt"
    },

    # 科技相关查询
    "Python 编程": {
        "doc2.txt", "doc4.txt", "doc8.txt", "doc10.txt", "doc14.txt", "doc18.txt", "doc20.txt"
    },
    "机器学习": {
        "doc2.txt", "doc4.txt", "doc5.txt", "doc6.txt", "doc11.txt", "doc15.txt", "doc19.txt"
    },
    "大数据": {
        "doc4.txt", "doc6.txt", "doc8.txt", "doc10.txt", "doc14.txt", "doc17.txt", "doc21.txt"
    },

    # 经济相关查询
    "北京 经济": {
        "doc3.txt", "doc5.txt", "doc9.txt", "doc11.txt", "doc12.txt", "doc15.txt", "doc19.txt"
    },
    "上海 金融": {
        "doc5.txt", "doc7.txt", "doc12.txt", "doc13.txt", "doc16.txt", "doc20.txt", "doc22.txt"
    },

    # 测试查询
    "浙江大学": {
        "doc4.txt", "doc7.txt", "doc10.txt", "doc14.txt", "doc16.txt", "doc18.txt", "doc21.txt"
    },
    "华为 公司": {
        "doc5.txt", "doc8.txt", "doc11.txt", "doc15.txt", "doc17.txt", "doc19.txt", "doc22.txt"
    }
}

# 测试查询集
TEST_QUERIES = [
    "北京 大学",
    "清华大学",
    "计算机 科学",
    "人工智能",
    "上海 大学",
    "Python 编程",
    "机器学习",
    "大数据",
    "北京 经济",
    "上海 金融",
    "浙江大学",
    "华为 公司"
]

# 查询分类（用于分析）
QUERY_CATEGORIES = {
    "教育": ["北京 大学", "清华大学", "上海 大学", "浙江大学"],
    "科技": ["计算机 科学", "人工智能", "Python 编程", "机器学习", "大数据"],
    "经济": ["北京 经济", "上海 金融"],
    "企业": ["华为 公司"]
}