import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import os
import threading
from datetime import datetime, date, timedelta
from typing import Optional


class DailyFileHandler(logging.Handler):
    """
    启动即写入 app-YYYY-MM-DD.log
    跨天自动切换文件，无 app.log 基准文件
    可选：保留最近 backup_days 天，自动清理旧日志
    """

    def __init__(
        self,
        log_dir: str | Path,
        prefix: str = "app",
        encoding: str = "utf-8",
        backup_days: int = 14,
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.prefix = prefix
        self.encoding = encoding
        self.backup_days = backup_days

        self._lock = threading.RLock()
        self._current_date: Optional[date] = None
        self._stream = None  # type: ignore

        # 启动时立即打开“今天”的文件
        self._open_for_today()

    def _filename_for(self, d: date) -> Path:
        return self.log_dir / f"{self.prefix}-{d:%Y-%m-%d}.log"

    def _open_for_today(self) -> None:
        today = date.today()
        self._open_for_date(today)

    def _open_for_date(self, d: date) -> None:
        # 关闭旧 stream
        if self._stream:
            try:
                self._stream.flush()
            finally:
                self._stream.close()

        self._current_date = d
        path = self._filename_for(d)
        self._stream = open(path, mode="a", encoding=self.encoding)

        # 每次切换文件时做一次清理
        self._cleanup_old_files()

    def _should_rollover(self) -> bool:
        return self._current_date != date.today()

    def _cleanup_old_files(self) -> None:
        if not self.backup_days or self.backup_days <= 0:
            return

        cutoff = date.today() - timedelta(days=self.backup_days)

        # 删除 log_dir 下符合 prefix-YYYY-MM-DD.log 且日期 < cutoff 的文件
        for p in self.log_dir.glob(f"{self.prefix}-????-??-??.log"):
            name = p.name  # app-2026-02-07.log
            try:
                ds = name[len(self.prefix) + 1 : len(self.prefix) + 11]  # YYYY-MM-DD
                file_date = datetime.strptime(ds, "%Y-%m-%d").date()
            except Exception:
                continue

            if file_date < cutoff:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    # 清理失败不影响主流程
                    pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                if self._should_rollover():
                    self._open_for_today()
                self._stream.write(msg + "\n")  # type: ignore
                self._stream.flush()            # 实时落盘（需要更高性能可改成按需 flush）
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream:
                try:
                    self._stream.flush()
                finally:
                    self._stream.close()
                self._stream = None
        super().close()

import logging
import sys
from pathlib import Path


def _get_log_dir() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "logs"


def setup_root_logger(log_dir: str | None = None) -> None:
    log_dir_path = Path(log_dir) if log_dir else _get_log_dir()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(logging.DEBUG)

    formatter_console = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter_file = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(process)d | %(threadName)s | "
            "%(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter_console)

    # 文件：启动即 app-YYYY-MM-DD.log
    file_handler = DailyFileHandler(
        log_dir=log_dir_path,
        prefix="app",
        backup_days=14,            # 保留14天
        encoding="utf-8",
        level=logging.DEBUG,
    )
    file_handler.setFormatter(formatter_file)

    root.addHandler(console_handler)
    root.addHandler(file_handler)


if __name__ == "__main__":
    setup_root_logger()
    logger = logging.getLogger("test_logger")
    logger.info("这是一条测试日志，应该同时出现在控制台和文件中")

    import os

    log_path = _get_log_dir()
    if log_path.exists():
        print(f"✅ 成功！文件已生成，大小: {os.path.getsize(log_path)} bytes")
    else:
        print("❌ 失败！文件仍未生成，请检查文件夹权限。")
