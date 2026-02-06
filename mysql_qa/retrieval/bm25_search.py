import logging

from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理

from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.utils.preprocess import preprocess_text

logger = logging.getLogger(__name__)


class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化分词后的问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        # 加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取问题
            self.original_questions = self.mysql_client.fetch_questions()
            if not self.original_questions:
                # 记录无问题警告
                self.logger.warning("未加载到问题")
                return
            # 分词问题
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]
            # 存储原始问题到 Redis
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)
        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        # self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        # 搜索查询
        if not query or not isinstance(query, str):
            # 记录无效查询
            self.logger.warning("无效查询")
            # 返回 None 和 False
            return None, False
        # 检查 Redis 缓存
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            # 返回缓存答案
            logger.info(f'命中缓存 --> answer:{query} <--')
            return cached_answer, False
        try:
            # 分词查询
            query_tokens = preprocess_text(query)
            # 计算 BM25 分数
            scores = self.bm25.get_scores(query_tokens)
            # 计算 Softmax 分数
            softmax_scores = self._softmax(scores)
            # 获取最高分索引
            best_idx = softmax_scores.argmax()
            # 获取最高分
            best_score = softmax_scores[best_idx]
            # 检查是否超过阈值
            if best_score >= threshold:
                # 获取原始问题
                original_question = self.original_questions[best_idx]
                # 获取答案
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 缓存答案
                    self.redis_client.set_data_ex(f"answer:{query}", answer)
                    # 记录搜索成功
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    return answer, False
            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            return None, True
        except Exception as e:
            # 记录搜索失败
            self.logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True


if __name__ == '__main__':
    from base.logger import setup_root_logger

    setup_root_logger()

    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    q = 'windows如何安装redis'
    q = 'windows如何安装mysql'
    logger.info(f'query: {q}')
    search_rst = bm25_search.search(q)
    show_str_num = 50
    if search_rst:
        logger.info(f'answer: {search_rst[0][:show_str_num]}')
    else:
        logger.warning(f'answer: {search_rst[0][:show_str_num]}')
