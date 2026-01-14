import os
import jieba
from config.settings import DATA_DIR


class DataLoader:
    def __init__(self, preprocess=True):
        self.preprocess = preprocess
        self.documents = {}
        self.load_data()

    def load_data(self):
        """加载并预处理文档数据"""
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")

        # 创建一些示例数据文件（如果data文件夹为空）
        if not os.listdir(DATA_DIR):
            self._create_sample_data()

        for file in os.listdir(DATA_DIR):
            if file.endswith(".txt"):
                path = os.path.join(DATA_DIR, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                    if self.preprocess:
                        # 基础文本预处理
                        content = self.preprocess_text(content)

                    self.documents[file] = content

        print(f"成功加载 {len(self.documents)} 个文档")

    def _create_sample_data(self):
        """创建示例数据文件"""
        sample_docs = {
            "doc1.txt": "清华大学是北京最好的大学之一，位于北京市海淀区。",
            "doc2.txt": "北京大学的计算机科学专业在全国排名前列。",
            "doc3.txt": "北京有多所顶尖大学，包括清华大学和北京大学。",
            "doc4.txt": "浙江大学在杭州，计算机科学专业也很强。",
            "doc5.txt": "2024年大学排名显示，北京的高校表现优异。",
            "doc6.txt": "上海交通大学在工程教育方面有很高的声誉。",
            "doc7.txt": "中国的工程教育质量在不断提升，北京高校领先。",
            "doc8.txt": "最新大学排名中，北京的多所高校进入前100名。"
        }

        for filename, content in sample_docs.items():
            path = os.path.join(DATA_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        print("已创建示例数据文件")

    def preprocess_text(self, text):
        """文本预处理"""
        # 去除多余空白字符
        text = ' '.join(text.split())
        return text

    def get_document(self, doc_id):
        """根据ID获取文档"""
        return self.documents.get(doc_id)

    def get_all_documents(self):
        """获取所有文档"""
        return list(self.documents.items())

    def get_document_ids(self):
        """获取所有文档ID"""
        return list(self.documents.keys())

    def tokenize(self, text):
        """分词处理"""
        return list(jieba.cut(text.lower()))