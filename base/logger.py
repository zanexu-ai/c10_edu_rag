import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def _get_absolute_log_path() -> Path:
    # 使用绝对路径：当前文件所在目录的父目录的 logs 文件夹
    # 假设该脚本在 project/utils/logger.py
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    return log_dir / "app.log"


def setup_root_logger(log_file: str | None = None) -> None:
    # 1. 确定路径
    full_path = Path(log_file) if log_file else _get_absolute_log_path()
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # 打印一下，确保你知道日志到底去哪了
    # print(f"--- 日志将保存至: {full_path.absolute()} ---")

    root = logging.getLogger()

    # 2. 关键：清除现有的 Handler，防止配置被拦截
    if root.hasHandlers():
        root.handlers.clear()

    # 3. 设置 Root 级别（必须为最低级别，Handler 才能过滤）
    root.setLevel(logging.DEBUG)

    formatter_console = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    formatter_file = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(process)d | %(threadName)s | "
            "%(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. 控制台 Handler (输出到 stderr 或 stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter_console)

    # 5. 文件 Handler
    file_handler = TimedRotatingFileHandler(
        filename=str(full_path),
        when="D",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        delay=True  # 只有在产生日志时才真正创建文件
    )
    file_handler.setLevel(logging.DEBUG)  # 设置为 DEBUG 确保捕获所有细节
    file_handler.setFormatter(formatter_file)

    root.addHandler(console_handler)
    root.addHandler(file_handler)


if __name__ == "__main__":
    setup_root_logger()
    logger = logging.getLogger("test_logger")
    logger.info("这是一条测试日志，应该同时出现在控制台和文件中")

    import os

    log_path = _get_absolute_log_path()
    if log_path.exists():
        print(f"✅ 成功！文件已生成，大小: {os.path.getsize(log_path)} bytes")
    else:
        print("❌ 失败！文件仍未生成，请检查文件夹权限。")
