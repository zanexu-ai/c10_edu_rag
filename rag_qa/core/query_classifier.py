# 导入标准库
import logging

import json
import os
# 导入 PyTorch
import torch
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from base.config import project_root

logger = logging.getLogger(__name__)
Trainer
"""
有什么:
1:有小数据集 
2:有模型bert-base-chinese
3:需要对上面的模型按照需求进行微调(训练)   _ 768 ->2分类
4:交付  bert_query_classifier
"""

rag_qa_path = os.path.join(project_root, 'rag_qa')


# todo 1 定义queryclassfier类型
# 作用: 封装bert查询分类的完整流程 (模型加载,模型训练,模型评估,模型预测)
class QueryClassifier:
    # 1.初始化方法: 配置模型路径 ,加载分词器,选择设备,定义映射标签
    def __init__(self, model_path="../models/bert_query_classifier"):
        # 1.存储模型路径:用于后续模型保存
        self.model_path = model_path
        # 2.加载分词器bert,将文本转成模型可识别的tokenid
        # 2.1拼接预训练模型bert的本地路径
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # 3.初始化模型变量
        self.model = None
        # 4.选择运行设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 5.日志模型记录运行信息
        logger.info(f"模型运行设备:{self.device}")
        # 6.定义标签映射 ,将文本标签转为数字 便于模型训练
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 7.加载模型:初始化时,自动调用load_model ,确保模型可用
        self.load_model()

    # 2.加载模型方法:从指定路径加载已经训练模型,若不存在则初始化模型
    def load_model(self):
        # 判断模型路径是否存在(即:是否已训练模型)
        if os.path.exists(self.model_path):  # ../models/bert_query_classifier
            # 1.1加载已训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 1.2将模型移动到指定设备
            self.model.to(self.device)
            # 1.3日志记录加载成功
            logger.info(f"模型加载成功:{self.model_path}")
        else:
            # 2.若模型路径不存在,初始化新模型
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese",
                                                                       num_labels=len(self.label_map))
            # 1.2将模型移动到指定设备
            self.model.to(self.device)
            logger.info(f"模型加载成功:初始化新的bert模型")

    # 5.创建符合要求的数据集
    def create_dataset(self, encodings, labels):
        # 1.定义内部类Dataset 集成pytorch的dataset
        class DataSet(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                # 初始化父类信息
                super().__init__()
                # 文本编码
                self.encodings = encodings
                # 标签
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            # 获取数据集的某一个元素
            def __getitem__(self, idx):
                # 提取第idx条编码数据
                # self.encodings.itmes() 字典遍历获取键和值
                item = {key: val[idx] for key, val in self.encodings.itmes()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        return DataSet(encodings, labels)

    # 4. 数据预处理
    def preprocess_data(self, texts, labels):
        """
        对文本进行分词,截断,填充,将标签转换为数字
        :param text:  待处理文本列表
        :param labels:  文本对应的标签列表(通用知识,专业知识)
        :return: 处理后的编码
        """
        # 1.文本编码
        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        # 2.标签转换
        labels = [self.label_map[label] for label in labels]

        # 3.返回结果
        return encodings, labels

    # 6. 训练模型方法: 加载数据集 ,数据预处理,配置训练参数,并且训练模型
    def train_model(self, data_file="../classify_data/model_generic_5000.json"):
        """
        该函数用于训练bert分类模型 ,区分查询类型 : 通用知识  和 专业知识
        :param data_file: 预训练数据文件
        :return:  None 因为训练好的模型直接保存起来的
        """
        # 1.检查数据集文件是否存在
        if not os.path.exists(data_file):
            logger.error(f"数据集文件{data_file}不存在")
            raise FileNotFoundError(f"数据集文件{data_file}不存在!")
        # 2. 加载数据集:从json文件中读取查询文件和对应的标签
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f.readlines()]

        # 3.提取文本和标签  :
        texts = [item['query'] for item in data]
        # print(texts)  #['什么是Skip List（跳表）？它的时间复杂度是多少？'.....]
        labels = [item['label'] for item in data]  # ['通用知识', '专业咨询'.....]
        # print(labels)

        # 4.划分训练集和验证
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2,
                                                                            random_state=18)

        # print(train_tests)
        # print(train_labels)
        #

        # 5.数据预处理: 将文本转换为bert输入格式(分词,截断,填充),标签转换为数字
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        logger.info(f'val_encodings:{val_encodings.keys()}')  # ['input_ids', 'token_type_ids', 'attention_mask']
        logger.info(f'val_labels:{val_labels}')  # [1, 1, 0, 1, 1, 1, ]

        # 6.创建数据集对象:将编码和标签封装pytorch的dataset对象
        self.create_dataset(train_encodings, train_labels)
        self.create_dataset(val_encodings, val_labels)

    # 7.评估模型
    def compute_metrics(self):
        pass

    # 3.保存模型
    def sava_model(self):
        pass


if __name__ == "__main__":
    from base.logger import setup_root_logger

    setup_root_logger()

    logger.info(f'rag_qa_path: {rag_qa_path}')

    query_classifier = QueryClassifier()
    query_classifier.train_model()
