# 1.导包  导入配置文件并且解析
import configparser  # 导入配置文件解析库,用于解析ini格式的配置文件
import os  # 导入操作系统库,用户获取当前文件所在路径

# todo 1 配置文件路径
# 1.1 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 1.2获取当前文件所在目录的绝对路径
current_dir_path = os.path.dirname(current_file_path)
# 1.3获取项目根路径的绝对路径
project_root = os.path.dirname(current_dir_path)
# 1.4拼接配置文件(config.ini)的完整路径
config_file_path = os.path.join(project_root, 'config.ini')


# todo 2 解析配置文件(封装配置文件的读取逻辑,提供统一配置文件参数访问接口)
# 目标:你给我传入配置文件,我来解析你的配置文件
class Config:
    # 2.1初始化方法:读取配置文件并解析各个服务的配置参数
    # 参数解释:config_file_path   配置文件的完整路径(包含文件)
    def __init__(self, config_file=config_file_path):
        # 1.创建配置文件解析器
        self.config = configparser.ConfigParser()
        # 2.读取配置文件
        self.config.read(config_file)
        # 3.解析并存储各个服务器配置文件
        # 3.1解析MySQL数据库配置文件
        # 参1:要读取配置文件的模块  参2:要对的配置文件的参数名  参3:默认值
        # self.MYSQL_HOST = self.config.get("mysql", "host", fallback='localhost')
        self.MYSQL_HOST = self.config.get("mysql", "host", )
        self.MYSQL_PORT = self.config.get("mysql", "port", )
        self.MYSQL_USER = self.config.get("mysql", 'user', )
        self.MYSQL_PASSWORD = self.config.get("mysql", 'password', )
        self.MYSQL_DATABASE = self.config.get("mysql", 'database', )

        # 3.2 Redis 配置
        self.REDIS_HOST = self.config.get('redis', 'host', )
        self.REDIS_PORT = self.config.getint('redis', 'port', )
        self.REDIS_PASSWORD = self.config.get('redis', 'password', )
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', )
        self.REDIS_CACHE_SECONDS = self.config.getint('redis', 'cache_seconds', )

        # 3.3 日志 文件路径
        self.LOG_FILE = self.config.get('logger', 'log_file', )

        # 3.4milvus配置
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')


conf = Config(config_file_path)

if __name__ == '__main__':
    print()
    print(project_root)
    print()
    print(conf.MYSQL_HOST)
    print(conf.MYSQL_PASSWORD)
    print(conf.MYSQL_USER)
    print(conf.MYSQL_DATABASE)
    print()
    print(conf.REDIS_HOST)
    print(conf.REDIS_PORT)
    print()
    print(conf.LOG_FILE)

    print(conf.LLM_MODEL)
    print(conf.PARENT_CHUNK_SIZE)
    print(conf.VALID_SOURCES)
