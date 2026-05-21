import argparse
import logging
import os
import time

try:
    from loguru import logger as _loguru_logger
except ImportError:
    _loguru_logger = None


class _StdLoggerAdapter:
    def __init__(self) -> None:
        self._configured = False
        self._logger = logging.getLogger("MTMatch")

    def add(self, log_path: str) -> None:
        if self._configured:
            return
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._configured = True

    def info(self, msg: str) -> None:
        if not self._configured:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
            self._configured = True
        self._logger.info(msg)


logger = _loguru_logger if _loguru_logger is not None else _StdLoggerAdapter()


def init_logger(file_name: str) -> str:
    package_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(package_root, "outputs", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{file_name}.log")
    logger.add(log_path)
    return log_path


def log(msg: str) -> None:
    logger.info(msg)


def log_args(args: argparse.Namespace) -> None:
    for k, v in args.__dict__.items():
        if not str(k).startswith("__"):
            log(f"{k}: {v}")
