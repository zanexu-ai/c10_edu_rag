import logging

from datetime import datetime

from pprint import pprint

import os, sys  # os 文本操作  sys:系统操作
from langchain_community.document_loaders import TextLoader  # 加载纯文本工具
# 加载markdown 文件  ,保留标题 列表名称等结构化信息
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 针对markdown文档文本进行切分, 尊重其结构化格式(例如:标题层级)
from langchain.text_splitter import MarkdownTextSplitter

from base.config import conf
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

logger = logging.getLogger(__name__)

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    """
    从directory_path（及子目录）中加载所有支持类型的文件，为每个文档添加元数据
    :param directory_path:  eg. ../data/ai_data/
    :return:  加载完成的文档列表（每个元素为langchain Document对象，包含page_content, metadata
    """
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    source = os.path.basename(directory_path).replace("_data", "")
    # print(f'supported_extensions: {supported_extensions}')
    # print(f'source: {source}')

    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # print(f'*'*99)
            # print(f'file_path: {file_path}')
            # print(f'*'*99)
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    # 根据文件扩展名获取对应的加载器类
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # print(f'loaded_docs: {loaded_docs}')
                    # print(f'len: {len(loaded_docs)}')
                    # for i, doc in enumerate(loaded_docs):
                    #     print(f'{i}: {doc}')
                    # 遍历加载的每个文档
                    for doc in loaded_docs:
                        # 为文档添加学科类别元数据
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # 将加载的文档添加到总列表中
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                    # 捕获加载过程中可能出现的异常
                except Exception as e:
                    # 记录加载失败的日志，包含错误信息
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                    # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents


# todo 3# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    """
    加载文档并且进行分层切块,先切成大粒度父块,在将父块切成小粒度子块,子块关联父块
    :param directory_path:文档目录路径
    :param parent_chunk_size: 父块切分长度(大粒度)
    :param child_chunk_size:  子块切分长度(小粒度)
    :param chunk_overlap:     子块切分重叠长度(确保上下文关联)
    :return:                  带元数据的子块列表(每个子块包含父块关联信息)
    """
    # 1.调用加载函数,从目录中加载所有文档
    # 2.初始化切分器 : 区分通用文档和markdown文档
    # 2.1通用文档切分
    # 2.1.1 初始化父块和子块分词器(通用),适合txt/pdf/word等 基于中文语义进行分割
    # 2.1.2 初始化markdown专用分词器,基于markdown语法结构分割.例如: 按照标题,段落
    # 3.初始化子块列表 ,存储最终切分结果
    child_chunks = []
    # 4.遍历每个原始文档(带索引i,用于生成唯一id)
    # 4.1获取文档的拓展名(判断是否为markdown,选择对应切分器)
    # 4.2根据文件类型选择切分器
    # 4.3 第1步:将文档切分成大块(绕)
    # 5.遍历每个父块(带索引j,用于生成唯一id)
    # 5.1生成父块唯一ID(格式:doc_文档索引_parent_父块索引)
    # 5.2 为父块添加元数据(用于后续rag检索溯源) parent_id,parent_content
    # 5.3 第2步:将父块切分成小块(子块)  小粒度 , 便于精准匹配
    # 6.遍历每个子块(带索引k,用于生成唯一id)
    # 6.1 为子块添加关联父块的元数据 parent_id,parent_content
    # 6.2 生成子块唯一的id (格式: 父块ID_child_子块索引)
    # 6.3 将子块添加到子块列表中
    # 7.记录切分后的字总块数
    # 8.返回子块列表(供后续向量存储和检索使用)
    return child_chunks


if __name__ == '__main__':
    from base.logger import setup_root_logger

    setup_root_logger()

    q = 'document_processor.py是EduRAG系统的核心模块之一，用于文档解析。主要负责加载多种格式的文档（如.txt、.pdf等），并对其进行分层切分，生成父块和子块，为后续的向量存储和检索做好准备。'
    q = '\n\n'.join([q * 100, q * 20])

    # ats = AliTextSplitter()
    # rst = ats.split_text(q)
    # pprint(rst)
    # print(len(rst))
    #
    # print(f'-' * 99)
    #
    # crts = ChineseRecursiveTextSplitter()
    # rst = crts.split_text(q)
    # pprint(rst)
    # print(len(rst))

    documents = load_documents_from_directory('../data/ai_data')
    logger.debug(f'-' * 99)
    for i, doc in enumerate(documents):
        logger.debug(f'{i}: {doc}')
        logger.debug(f'-' * 99)
    logger.debug(len(documents))

    # for root, dirs, files in os.walk("../data/ai_data"):
    #     print(f'root: {root}')  # 当前遍历到的目录
    #     print(f'dirs: {dirs}')  # 当前遍历到的目录
    #     print(f'files: {files}')  # 当前遍历到的目录
    #     print(f'*' * 99)
