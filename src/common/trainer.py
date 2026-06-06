import sys

sys.path.append("..")
import numpy as np
from src.common.models import IModel
from src.common.optimizers import IOptimizer
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
                self.clip_grad(gradients, max_grad)
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

    def clip_grad(self, grads: np.ndarray, max_norm):
        if max_norm is None:
            return
        total_norm = 0
        for grad in grads:
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)

        rate = max_norm / (total_norm + 1e-6)
        if rate < 1:
            for grad in grads:
                grad *= rate

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
