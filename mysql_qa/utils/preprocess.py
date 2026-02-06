# 1. 导入分词库
import logging

import jieba

logger = logging.getLogger(__name__)


# 预处理文本
def preprocess_text(text):
    # logger.info("开始预处理文本")
    try:
        # 分词并且转换为小写
        return jieba.lcut(text.lower())
    except Exception as e:
        logger.error(f"文本预处理失败:{e}")
        return []


if __name__ == '__main__':
    print(preprocess_text("黑马程序员ITCAST"))
