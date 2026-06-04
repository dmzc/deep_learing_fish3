import numpy as np
from abc import ABC, abstractmethod
from src.common.layers import ILayer, AffineLayer, SoftmaxWithLossLayer


class IModel(ABC):
    # 所有层的权重
    _weights: list[np.ndarray]

    # 所有层的梯度
    _gradients: list[np.ndarray]

    def __init__(self):
        self._weights = []
        self._gradients = []
        pass

    @abstractmethod
    def forward(self, contexts, target) -> float:
        """
        前向传播，返回损失函数值
        """
        pass

    @abstractmethod
    def backward(self, dout) -> None:
        """
        反向传播，不需要返回值
        """
        pass


class SimpleCbowModel(IModel):
    # 输入层矩阵
    _W_in: np.ndarray
    # 输出层矩阵
    _W_out: np.ndarray

    __in_layer1: ILayer
    __in_layer2: ILayer
    __out_layer: ILayer
    __loss_layer: ILayer

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        W_in = self._W_in = 0.01 * np.random.randn(vocab_size, hidden_size)
        W_out = self._W_out = 0.01 * np.random.randn(hidden_size, vocab_size)
        self.__in_layer1 = AffineLayer(W_in)
        self.__in_layer2 = AffineLayer(W_in)
        self.__out_layer = AffineLayer(W_out)
        self.__loss_layer = SoftmaxWithLossLayer()
        self._weights.extend(
            [self.__in_layer1.W, self.__in_layer2.W, self.__out_layer.W]
        )
        self._gradients.extend(
            [
                self.__in_layer1.weight_gradients,
                self.__in_layer2.weight_gradients,
                self.__out_layer.weight_gradients,
            ]
        )

    def forward(self, x: np.ndarray, t: np.ndarray):
        """
        contexts - (6, 2, 7)，6代表mini-batch数，2代表2个上下文词，7代表词表大小（one-hot变量）
        target - (6, 7)，则是6个样本的中间词变量
        """
        h0 = self.__in_layer1.forward(x[:, 0])
        h1 = self.__in_layer2.forward(x[:, 1])
        h = (h0 + h1) * 0.5
        score = self.__out_layer.forward(h)
        loss = self.__loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        ds = self.__loss_layer.backward(dout)
        da = self.__out_layer.backward(ds)
        da *= 0.5
        self.__in_layer1.backward(da)
        self.__in_layer2.backward(da)
