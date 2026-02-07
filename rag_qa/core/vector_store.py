# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
import logging

from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import os, sys, torch

# 配置项目路径: 确保程序能正确导入其它目录的模块.
# 1. 获取当前文件所在目录的绝对路径.
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取core文件所在的目录的绝对路径.
rag_qa_path = os.path.dirname(current_dir)
# sys.path.insert(0, rag_qa_path)
# 3. 获取项目根目录路径.
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from base.config import Config

logger = logging.getLogger(__name__)
# todo 1 初始化全局配置
conf = Config()


# def get_milvus_client(db_name):
#     # 如果uri为数据库名称路径，代表本地操作数据库
#     # client = MilvusClient(uri="milvus_demo.db")
#     # 如果uri为链接地址，代表Milvus属于单机服务，需要开启Milvus后台服务操作
#     client = MilvusClient(uri="http://localhost:19530")
#     # client = MilvusClient(uri="http://192.168.21.22:19530")
#     # # # 创建名称为milvus_demo的数据库
#     # #
#     databases = client.list_databases()
#     logger.info(f'milvus databases: {databases}')
#     if db_name not in databases:
#         client.create_database(db_name=db_name)
#     else:
#         client.using_database(db_name=db_name)
#     return client


# todo 2  定义VectorStore类 ,封装向量库的核心功能(集合管理,文档入库,混合搜索,结果处理)
class VectorStore:
    # 1.初始化类方法 :配置向量库连接 ,加载模型(bge-m3  重排序),初始化客户端
    def __init__(self, collection_name=conf.MILVUS_COLLECTION_NAME, host=conf.MILVUS_HOST, port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database
        self.logger = logger

        # 选择模型运行设备 优先使用GPU(cuda)加速,无GPU则用CPU,去pytorch安装
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f'device: {self.device}')

        # 初始化 BGE-Reranker模型
        # 拼接重排序模型本地路径
        # bge_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        # print(f"bge_path-------------:{bge_path}")
        # 加载本地模型
        # 参1: 模型本地路径  参2:GPU启用半精度计算(减少内存占用,提升速度)  参3:模型运行设备
        self.embedding_function = BGEM3EmbeddingFunction(model_name_or_path="../models/bge-reranker-large",
                                                         use_fp16=(self.device == "cuda"),
                                                         device=self.device)

        # 获取稠密向量维度
        self.dense_dim = self.embedding_function.dim['dense']
        logger.info(self.embedding_function.dim)

        # 初始化milvus客户端
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # collections=self.client.list_collections()
        # logger.info(f'collections: {collections}')
        self.get_milvus_client('itcast')

        # 调用方法创建或者加载milvus集合(类似于:建表,没有就建,有就加载)
        self._create_or_load_collection()

    def get_milvus_client(self, db_name):
        # 如果uri为数据库名称路径，代表本地操作数据库
        # client = MilvusClient(uri="milvus_demo.db")
        # 如果uri为链接地址，代表Milvus属于单机服务，需要开启Milvus后台服务操作
        # client = MilvusClient(uri="http://localhost:19530")
        # client = MilvusClient(uri="http://192.168.21.22:19530")
        # # # 创建名称为milvus_demo的数据库
        # #
        databases = self.client.list_databases()
        logger.info(f'milvus databases: {databases}')
        if db_name not in databases:
            self.client.create_database(db_name=db_name)
        else:
            self.client.using_database(db_name=db_name)
        return self.client

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # todo 2  将文档(子块)转换为向量并且存储milvus中
    def add_documents(self, documents):
        pass

    # todo 3 混合检索(稠密+稀疏)+重排序,返回精准父文档
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        pass

    # todo 4 从子块列表中提取去重父文档
    def _get_unique_parent_docs(self, sub_chunks):
        pass


if __name__ == '__main__':
    from base.logger import setup_root_logger

    setup_root_logger()

    vector_store = VectorStore()
