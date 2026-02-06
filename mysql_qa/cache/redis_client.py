import redis
import json
import os, sys

import logging

from base.config import conf

logger = logging.getLogger(__name__)


# 2.定义redis客户端类->封装redis连接,存储数据,数据获取(key),答案查询等功能
class RedisClient:
    def __init__(self):
        # 1.定义日志记录器
        self.logger = logger

        # 2.建立redis客户端连接
        try:
            self.client = redis.StrictRedis(
                host=conf.REDIS_HOST,
                port=conf.REDIS_PORT,
                password=conf.REDIS_PASSWORD,
                db=conf.REDIS_DB,
                decode_responses=True  # 默认为True,获取数据时,返回字符串,否则返回字
            )
            # 3.记录info日志:redis连接成功
            # self.logger.info("Redis连接成功!")

        except Exception as e:
            self.logger.info(f"Redis连接失败!:{e}")
            raise

    # 1.2 存储数据到redis中
    def set_data(self, key, value):
        try:
            # 1.存储数据
            # 参1:要存储的键名(即:问题)  参2: 键值(即:答案)
            r = self.client.set(key, json.dumps(value))

            # 2.记录info级别日志:存储数据成功
            self.logger.info(f"存储数据到redis成功 --> {key}")
            return r

        except redis.RedisError as e:
            # 捕获redis异常
            self.logger.error(f"redis存储数据失败: {e}")

    def set_data_ex(self, key, value):
        try:
            # 1.存储数据
            # 参1:要存储的键名(即:问题)  参2: 键值(即:答案)
            r = self.client.setex(key, conf.REDIS_CACHE_SECONDS, json.dumps(value))

            # 2.记录info级别日志:存储数据成功
            self.logger.info(f"存储数据到redis成功 --> {key}")
            return r

        except redis.RedisError as e:
            # 捕获redis异常
            self.logger.error(f"redis存储数据失败: {e}")

    # 1.3从redis中获取数据
    def get_data(self, key):
        try:
            # 1.获取数据
            data = self.client.get(key)
            # 2.打印处理结果  若存在数据就用json.loads方法转换为字典,否则返回null
            # print(data)
            return json.loads(data) if data else None

        except Exception as e:
            self.logger.error(f"redis获取数据失败:{e}")
            return None

    # 1.4 根据查询内容从redis中获取缓存的答案
    def get_answer(self, query):
        result = self.get_data(f"answer:{query}")
        return result


if __name__ == '__main__':
    # 1.实例化RedisClient类
    redis_client = RedisClient()
    print(f"redis_client:{redis_client}")
    q = 'PyCharm使用中文版本还是英文版本4'
    a = '强烈推荐使用英文版本，中文版本虽然一时半会儿可以让你更快熟悉PyCharm这个软件的部分功能的使用，但是英文IT界内大部分人都是使用的英文版本，所以使用中文版本就会有沟通问题，当你碰到什么问题，或者想要学习更多新功能的时候，此时看到的资料大部分都是英文的。所以推荐刚刚开始学习的时候就使用英文版本，这时候如果有碰到英文单词不知道什么意思，建议用工具查一下并且记录下来，PyCharm软件常用的英文单词很少，而且含义单一，很好记忆'
    set_result = redis_client.set_data(f'answer:{q}', a)
    print(f'set_result:{set_result}')
    # 4.测试get_answer
    ans = redis_client.get_answer(query=q)
    print(f'ans: {ans}')
