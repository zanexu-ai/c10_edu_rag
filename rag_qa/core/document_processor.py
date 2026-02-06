from pprint import pprint

import os, sys  # os 文本操作  sys:系统操作
from langchain_community.document_loaders import TextLoader  # 加载纯文本工具
# 加载markdown 文件  ,保留标题 列表名称等结构化信息
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 针对markdown文档文本进行切分, 尊重其结构化格式(例如:标题层级)
from langchain.text_splitter import MarkdownTextSplitter

# todo1 配置文件路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# rag_qa_path = os.path.dirname(current_dir)
# sys.path.insert(0, rag_qa_path)
# project_root = os.path.dirname(rag_qa_path)
# sys.path.insert(0, project_root)
# print(f'sys.path: {sys.path}')

# 导入自定义包
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from rag_qa.edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter

if __name__ == '__main__':
    q = 'document_processor.py是EduRAG系统的核心模块之一，用于文档解析。主要负责加载多种格式的文档（如.txt、.pdf等），并对其进行分层切分，生成父块和子块，为后续的向量存储和检索做好准备。'
    q = '\n\n'.join([q * 100, q * 20])

    ats = AliTextSplitter()
    rst = ats.split_text(q)
    pprint(rst)
    print(len(rst))

    print(f'-' * 99)

    crts = ChineseRecursiveTextSplitter()
    rst = crts.split_text(q)
    pprint(rst)
    print(len(rst))
