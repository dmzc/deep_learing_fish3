import sys

sys.path.append("..")
import numpy as np
from src.common.decorators.timer import timer
from src.common.models import IModel
from src.common.optimizers import IOptimizer
from src.common.util import clip_grads
import matplotlib.pyplot as plt
import time as time


class Trainer:
    __model: IModel
    __optimizer: IOptimizer
    __ppl_list: list[int]
    __lost_list: list[int]

    def __init__(self, model: IModel, optimizer: IOptimizer):
        self.__model = model
        self.__optimizer = optimizer
        self.__ppl_list = []
        self.__lost_list = []

    @timer("训练一次耗时")
    def train(
        self,
        x: np.ndarray,
        t: np.ndarray,
        max_epoch=10,
        batch_size=32,
        max_grad=None,
        eval_interval=20,
    ):
        # 样本数
        data_size = len(x)

        # 一个epoch中要训练多少批（以确保一个epoch训练覆盖所有样本数）
        max_iters = data_size // batch_size

        model = self.__model
        optimizer = self.__optimizer

        total_loss = 0
        loss_count = 0
        for epoch in range(max_epoch):
            idxs = np.random.permutation(np.arange(data_size))  # [3, 1, 4, 2, 0]
            x = x[idxs]
            t = t[idxs]
            start_time = time.time()
            for iter in range(max_iters):
                batch_x = x[iter * batch_size : (iter + 1) * batch_size]
                batch_t = t[iter * batch_size : (iter + 1) * batch_size]
                loss = model.forward(batch_x, batch_t)
                total_loss += loss
                loss_count += 1
                model.backward()
                (weights, gradients) = model.get_weights_gradients()
                clip_grads(gradients, max_grad)
                optimizer.update(weights, gradients)
                # 用平均交叉熵损失计算困惑度
                if (eval_interval is not None) and (iter % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    ppl = np.exp(avg_loss)
                    self.__lost_list.append(float(avg_loss))
                    self.__ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
                # print(f"耗时{time.time() - start_time}")
                # a = 12
            # 一次epoch需要0.23秒
            print(f"训练epoch耗时:{time.time() - start_time}")

    def plot(self, ylim=None, use_ppl=False):
        data = self.__ppl_list if use_ppl else self.__lost_list
        x = np.arange(len(data))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, data, label="train")
        plt.xlabel("iterations")
        if use_ppl:
            plt.ylabel("ppl")
        else:
            plt.ylabel("loss")
        plt.show()


class RNNTrainer:
    __model: IModel
    __optimizer: IOptimizer
    __ppl_list: list[int]
    __eval_interval: int

    def __init__(self, model: IModel, optimizer: IOptimizer):
        self.__model = model
        self.__optimizer = optimizer
        self.__ppl_list = []

    def train(
        self,
        xs,
        ts,
        max_epoch=10,
        batch_size=20,
        time_size=35,
        max_grad=None,
        eval_interval=20,
    ):
        self.__eval_interval = eval_interval
        data_size = len(xs)
        # max_iters = data_size // (batch_size * time_size)
        max_iters = 10
        start_time = time.time()
        offset = 0
        model = self.__model
        optimizer = self.__optimizer
        total_loss = 0
        loss_count = 0
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch_data(
                    xs,
                    ts,
                    batch_size=batch_size,
                    time_size=time_size,
                    current_index=offset,
                )
                offset += 1
                loss = model.forward(batch_x, batch_t)
                model.backward()
                weights_gradients: tuple[list[np.ndarray], list[np.ndarray]] = (
                    model.get_weights_gradients()
                )
                weights = weights_gradients[0]
                gradients = weights_gradients[1]
                clip_grads(grads=gradients, max_grad=max_grad)
                optimizer.update(weights, gradients)
                total_loss += loss
                loss_count += 1
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                        % (
                            epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            ppl,
                        )
                    )
                    self.__ppl_list.append(ppl)
                    total_loss, loss_count = 0, 0

    def get_batch_data(
        self,
        xs: np.ndarray,
        ts: np.ndarray,
        batch_size: int,
        time_size: int,
        current_index: int,
    ):
        batch_x = np.empty((batch_size, time_size), dtype=np.int32)
        batch_t = np.empty((batch_size, time_size), dtype=np.int32)

        data_size = len(xs)
        jump = data_size // batch_size
        offsets = [
            i * jump for i in range(batch_size)
        ]  # mini-batch的各笔样本数据的开始位置
        time_index = current_index * time_size
        for t in range(time_size):
            for index, offset in enumerate(offsets):
                batch_x[index, t] = xs[(offset + time_index) % data_size]
                batch_t[index, t] = ts[(offset + time_index) % data_size]
            time_index += 1
        return batch_x, batch_t

    def plot(self, ylim=None):
        x = np.arange(len(self.__ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.__ppl_list, label="train")
        plt.xlabel("iterations (x" + str(self.__eval_interval) + ")")
        plt.ylabel("perplexity")
        plt.show()
