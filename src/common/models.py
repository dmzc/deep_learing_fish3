import numpy as np
from abc import ABC, abstractmethod
from src.common.layers import (
    ILayer,
    AffineLayer,
    SoftmaxLossLayer,
    EmbeddingLayer,
    NegativeSamplingLossLayer,
)
from src.common.vocab import Vocab
from src.common.decorators.timer import timer


class IModel(ABC):
    # 所有层的权重
    _weights: list[np.ndarray]

    # 所有层的梯度
    _gradients: list[np.ndarray]

    def __init__(self):
        self._weights = []
        self._gradients = []
        pass

    def get_weights_gradients():
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
        W_in = self._W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype(
            np.float32
        )
        W_out = self._W_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype(
            np.float32
        )
        self.__in_layer1 = AffineLayer(W_in)
        self.__in_layer2 = AffineLayer(W_in)
        self.__out_layer = AffineLayer(W_out)
        self.__loss_layer = SoftmaxLossLayer()
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

    def get_weights_gradients(self):
        model = self
        gradients = model._gradients[:]
        weights = model._weights[:]
        oIndex = 0
        while oIndex < len(weights):
            oWeight = weights[oIndex]
            oGradient = gradients[oIndex]
            shared_indexs: list[int] = []
            for iIndex in range(oIndex + 1, len(weights)):
                if oWeight is weights[iIndex]:
                    oGradient += gradients[iIndex]
                    shared_indexs.append(iIndex)
            for index in reversed(shared_indexs):
                weights.pop(index)
                gradients.pop(index)
            oIndex += 1
        return (weights, gradients)

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


class CbowModel(IModel):
    # 输入层矩阵
    _W_in: np.ndarray
    # 输出层矩阵
    _W_out: np.ndarray

    _G_in: list[np.ndarray]

    _G_out: list[np.ndarray]

    _in_layers: list[ILayer]

    _loss_layer: ILayer

    def __init__(self, vocab: Vocab, hidden_size: int):
        super().__init__()
        vocab_size = vocab.get_size()
        window_size = vocab.get_current_window_size()
        # 使用加速的方法替代矩阵乘法，此处W_in为词作为上下文词时的矩阵，W_out为词作为中心词时的矩阵，所以两者含义是截然不同的
        W_in = self._W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype(
            np.float32
        )
        W_out = self._W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype(
            np.float32
        )
        in_layers = self._in_layers = []
        self._G_in = in_layers_gradients = []
        for index in range(window_size * 2):
            in_layer = EmbeddingLayer(W_in)
            in_layers.append(in_layer)
            in_layers_gradients.append(in_layer.weight_gradients)

        loss_layer = self._loss_layer = NegativeSamplingLossLayer(
            vocab=vocab, out_matrix=W_out
        )
        # 此loss layer，内部有多个负采样的embed layer
        self._G_out = loss_layer.weight_gradients

    def get_weights_gradients(self):
        return (
            [self._W_in, self._W_out],
            [np.sum(self._G_in), np.sum(self._G_out)],
        )

    def forward(self, contexts: np.ndarray, target: np.ndarray):
        in_vec = 0
        in_layers = self._in_layers
        for index, layer in enumerate(in_layers):
            in_vec += layer.forward(contexts[:, index])
        in_vec *= 1 / len(in_layers)
        loss = self._loss_layer.forward(in_vec, target)
        return loss

    def backward(self, dout=1):
        dout = self._loss_layer.backward(dout)
        dout *= 1 / len(self._in_layers)
        for layer in self._in_layers:
            layer.backward(dout)
