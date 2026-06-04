import time
from src.common.logger import getLogger


# 第一层：接收装饰器自己的参数
def timer(message: str):
    # 第二层：接收被装饰的函数
    def decorator(func):

        # 第三层：接收函数调用时的位置/关键字参数
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            ret = func(*args, **kwargs)
            cost = time.perf_counter() - start
            getLogger().info(f"{message}:{cost:.4f}")
            return ret

        return wrapper

    return decorator
