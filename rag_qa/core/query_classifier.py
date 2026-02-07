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
        #判断模型路径是否存在(即:是否已训练模型)
        if os.path.exists(self.model_path):  #../models/bert_query_classifier
            #1.1加载已训练模型
            self.model=BertForSequenceClassification.from_pretrained(self.model_path)
            #1.2将模型移动到指定设备
            self.model.to(self.device)
            #1.3日志记录加载成功
            logger.info(f"模型加载成功:{self.model_path}")
        else:
            #2.若模型路径不存在,初始化新模型
            self.model=BertForSequenceClassification.from_pretrained("../models/bert-base-chinese",num_labels=len(self.label_map))
            # 1.2将模型移动到指定设备
            self.model.to(self.device)
            logger.info(f"模型加载成功:初始化新的bert模型")

    # 5.创建符合要求的数据集
    def create_dataset(self, encodings, labels):
        pass

    # 4. 数据预处理
    def preprocess_data(self, text, labels):
        pass

    # 6. 训练模型方法: 加载数据集 ,数据预处理,配置训练参数,并且训练模型
    def train_model(self, data_file="../classify_data/model_generic_5000.json"):
        pass

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
