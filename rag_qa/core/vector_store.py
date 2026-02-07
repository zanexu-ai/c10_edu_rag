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

from rag_qa.core.document_processor import process_documents

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
        bge_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        # print(f"bge_path-------------:{bge_path}")
        # 加载本地模型
        # 参1: 模型本地路径  参2:GPU启用半精度计算(减少内存占用,提升速度)  参3:模型运行设备
        self.embedding_function = BGEM3EmbeddingFunction(
            # model_name_or_path="../models/bge-reranker-large",
            model_name_or_path=bge_path,
            use_fp16=(self.device == "cuda"),
            device=self.device, )

        # 获取稠密向量维度
        self.dense_dim = self.embedding_function.dim['dense']
        logger.info(self.embedding_function.dim)

        # 4.初始化bge-reranker 模型   用于重排序检索结果
        # 4.1拼接重排序模型本地路径    rag_qa\models\bge-reranker-large
        reranker_path = os.path.join(rag_qa_path, "models", "bge-reranker-large")
        # 4.2加载模型
        self.reranker = CrossEncoder(reranker_path, device=self.device)

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

    # todo 2  将文档(子块)转换为向量(稀疏_稠密的)并且存储milvus中
    def add_documents(self, documents):
        # 1.提取所有文档的内容列表:即每个文档对象中提取page_content属性
        texts = [doc.page_content for doc in documents]
        # 2.使用BGE-M3模型将文档内容转换为向量(稀疏,稠密)
        # 输入:texts
        # 输出:字典 包含:dense稠密向量列表  spares稀疏向量列表
        embeddings = self.embedding_function(texts)
        # print(f"embedding:{embedding}")  #

        # 3.初始化空列表,存储待插入milvus数据(每个元素为一个数据字典)
        data = []
        # 4.遍历文档,组装插入数据
        for i, doc in enumerate(documents):
            # 4.1生成文档唯一ID:对文档内容进行MD5哈希,避免重复插入
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            # 4.2 处理稀疏向量
            sparse_vector = {}
            # 4.2.1  获取第i个文档的稀疏向量行
            # row=embeddings['sparse'].getrow(i)  #会产生大量的警告信息
            row = embeddings['sparse'][[i]]
            # 4.2.2  获取稀疏向量的非零值
            indices = row.indices  # 非零值的索引列表   [10,25,50]
            values = row.data  # 非零值的权重列表   [0.1,0.2,0.3]
            # 4.2.3  组装稀疏向量   即:{索引:权重}  ->{(10,0.1),(25,0.2),(50,0.3)}
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # ✅ 必须：Milvus SPARSE_FLOAT_VECTOR 不允许空行
            if not sparse_vector:
                logger.warning(
                    f"skip empty sparse vector: i={i}, id={text_hash}, "
                    f"text_len={len(doc.page_content)}, preview={doc.page_content[:80]!r}"
                )
                continue
            # 4.3组装单条插入的数据
            data.append({
                "id": text_hash,  # 唯一ID(MD5哈希值)
                "text": doc.page_content,  # 文档内容,
                "dense_vector": embeddings['dense'][i],  # 稠密向量(BGE-M3生成)
                "sparse_vector": sparse_vector,  # 稀疏向量(BGE-M3生成)
                "parent_id": doc.metadata['parent_id'],  # 父文档ID
                'parent_content': doc.metadata["parent_content"],  # 父文档内容
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        # 5.插入数据到milvus中
        if data:
            self.client.upsert(self.collection_name, data)
            logger.info(f"已插入{len(data)}条数据到集合中")

    # 3.4 混合(稠密+稀疏)检索+结果重排序+返回精准父文档
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        该函数用于执行混合检索(稠密+稀疏_向量) +结果重排序 返回精准父文档
        :param query: 用户查询的文本
        :param k: 混合检索返回的top-k子块的数量,默认从conf文件中获取
        :param source_filter:学科过滤条件,  例如:  "AI":仅检索该学科的文档,None不过滤
        :return:重排序后的top-M个父文档列表
        """
        # 1.生成查询嵌入向量  根据用户的提问获取稠密和稀疏向量
        query_embedding = self.embedding_function([query])
        # print(f"query_embedding：{query_embedding}")
        # 2.提取查询稠密的向量
        dense_query_vector = query_embedding["dense"][0]

        # 3.处理查询的稀疏向量(存储非零值)
        sparse_query_vector = {}
        # 3.1获取查询的稀疏向量行(仅1个查询,获取第0行即可)
        row = query_embedding['sparse'][[0]]
        # 3.2获取非零值索引和权重
        indices = row.indices  # 非零值索引列表
        values = row.data  # 非零值权重列表
        # 3.3组装稀疏向量字段:将索引与权重配对
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # 4.构建检索过滤表达式(过滤学科)
        filter_expr = f"source=='{source_filter}'" if source_filter else ""
        # 5.构建稠密向量的检索请求 定义稠密向量 ANN
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",  # 稠密向量字段名称
            param={"metric_type": "IP", "params": {"nprobe": 10}},  # 检索参数 内积相似度 聚类数
            limit=k,  # 检索top-K子块数量
            expr=filter_expr  # 应用过滤表达式(按照学科进行过滤)   过滤结果 要么是 "" 或者source==ai
        )
        # 6.构建稀疏向量的检索请求 定义稀疏向量 ANN
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )
        # 7.创建加权排序器WeightedRanker(稠密ANN权重,稀疏ANN权重)
        ranker = WeightedRanker(1.0, 0.7)  # 稠密向量侧重:语义相似度  稀疏向量侧重:关键词匹配
        # 8.执行混合检索 :  同时进行稠密 +稀疏向量检索  ,用加权排序器 返回Top-m父文档列表
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],  # 混合检索请求(稠密+稀疏)
            ranker=ranker,  # 加权排序器
            limit=k,
            output_fields=['text', 'parent_id', 'parent_content', 'source', 'timestamp']
        )[0]  # 因为返回的是列表 ,取第0个元素
        # print(f"-----------------:{results}")  # 列表  [hit,hit,hit]
        # print(f"$$$$$$$$$$$$$$$$$$:{type(results)}")  #
        # print(f"results:{len(results)}")  #1
        # for  hit  in results:
        #     print(f"hit-------------:{hit['entity']}")
        # 9.将检索结果转化为document对象  目的:统一格式 ,方便后续处理
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # 10.从子块中提取去重父文档: 避免同一个父块的多个字段重复返回
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # 11.重排序逻辑: 父文档数量<2 跳过重排序(无需优化),直接返回即可
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
        # 12.父文档数量>=2 执行重排序
        if parent_docs:
            # 12.1 构建 查询-文档 配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 12.2计算相关性得分
            scores = self.reranker.predict(pairs)
            # for i ,doc in sorted(zip(scores,parent_docs),reverse=True):
            #     print(f"i-----{i}-----{doc}")
            # 12.3按得分降序排序
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            # 13.若父文档为空(无检索结果),返回空列表
            ranked_parent_docs = []
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # todo 3.5 从子块列表中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 3.6 将milvus检索结果转换为langchain的document对象
    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )


if __name__ == '__main__':
    from base.logger import setup_root_logger

    setup_root_logger()

    vector_store = VectorStore()
    # directory_path = "../data/ai_data"
    # documents = process_documents(directory_path)
    #
    # vector_store.add_documents(documents)

    # 测试混合检索
    query = "windows如何安装redis"
    result = vector_store.hybrid_search_with_rerank(query=query, source_filter="ai")
    print(f"result:{result}")
    print(f"result:{len(result)}")
