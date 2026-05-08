from loguru import logger
import os
import sys

_loggger = None


def getLogger():
    global _loggger
    if _loggger is None:
        _loggger = logger
        # 建日志文件夹
        os.makedirs("logs", exist_ok=True)

        # 清空默认处理器
        _loggger.remove()

        # 控制台输出
        _loggger.add(
            sys.stderr, format="{time:HH:mm:ss} | {level} | {module}:{line} | {message}"
        )
        # 文件日志
        _loggger.add(
            "logs/nlp_project.log",
            rotation="1 day",
            retention="7 days",
            encoding="utf-8",
        )
    return _loggger
