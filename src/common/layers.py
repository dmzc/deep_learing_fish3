import numpy as np

from abc import abstractmethod, ABC

from src.common.util import softmax, cross_entropy

from src.common.vocab import Vocab

from src.common.samplers import ISampler, SimpleSampler


class ILayer(ABC):
    @abstractmethod
    def get_weights_gradients() -> tuple[list[np.ndarray], list[np.ndarray]]: ...

    @abstractmethod
    def forward(self, x: np.ndarray, train_flag=False) -> np.ndarray: ...

    @abstractmethod
    def backward(self, dout: np.ndarray | float) -> np.ndarray: ...


"""
    relu层
"""


class ReluLayer(ILayer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


"""
    sigmod层
"""


class SigmodLayer(ILayer):
    def __init__(self):
        self.out = None

    def forward(self, x, train_flag=False):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


"""
    仿射层
"""


class AffineLayer(ILayer):
    weights: np.ndarray
    weight_gradients: np.ndarray
    bias: np.ndarray
    bias_gradients: np.ndarray

    x: np.ndarray

    def __init__(self, W, b=None):
        """
        不是所有层都需要偏置，训练word2vec时就不需要
        """
        self.weights = W
        self.weight_gradients = np.zeros_like(W)
        if b is not None:
            self.bias = b
            self.bias_gradients = np.zeros_like(b)

    def forward(self, x):
        self.x = x
        malmul = np.dot(x, self.weights)
        if self.bias is None:
            return malmul
        return malmul + self.bias

    def backward(self, dout):
        # 计算dx是为了向前传播
        dx: np.ndarray = np.dot(dout, self.weights.T)
        self.weight_gradients[:] = np.dot(self.x.T, dout)
        if self.bias is not None:
            self.bias_gradients[:] = np.sum(dout, axis=0)
        return dx

    def get_weights_gradients(self):
        if self.bias is not None:
            return (
                [self.weights, self.bias],
                [self.weight_gradients, self.bias_gradients],
            )


class SoftmaxLossLayer(ILayer):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t, train_flag=False):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # t=[[1, 0, 0], [0, 1, 0], [0, 0 ,1]]
            # 最后交叉熵损失层是把所有类别的损失都加起来，反向传播求导求的是这一整批
            dx = (self.y - self.t) / batch_size
        else:  # t=[0, 1, 2]
            dx = self.y.copy()
            # 这里减1是因为把索引数组转换为独热标签矩阵
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        # 的损失对输出的导数，所以要取平均梯度往前传播
        return dx


class SigmoidLossLayer(ILayer):
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # sigmoid的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        # 原来：x = [0.8, 0.3, 0.6]，t = [1, 0, 0]，第一个是正例，剩下的都是负例
        # 现在：x = [ [0.2, 0.8], [0.6, 0.3], [0.4, 0.6] ]，t不变，此时就可以套
        # 用多分类交叉熵这个函数,t正好就是当前正解标签
        self.loss = cross_entropy(np.c_[1 - self.y, self.y], self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx


class DropoutLayer(ILayer):
    """
    Dropout层，用于防止过拟合，效果相当于集成学习中，多个优化器训然后平均的效
    果。
    训练时，随机将输入的一些元素设为0，从而减少模型对某些特征的依赖，防止过拟合。
    推理时，要将所有元素乘以(1 - dropout_ratio)，从而保持输出的期望不变。
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 推理时不丢弃，但是要缩放下尺度
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalizationLayer(ILayer):
    """
    批归一层，将每层输出做归一化，使得每次都尽量遵循同一分布。优点：

    可以使学习快速进行（增大学习率）
    不那么依赖初始值
    抑制过拟合

    TODO:这里的计算图要推导下，没有看的太懂

    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x: np.ndarray, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)  # 按列求n个样本各个特征的均值
            xc = x - mu  # 减去均值后的值
            var = np.mean(xc**2, axis=0)  # 方差
            std = np.sqrt(var + 10e-7)  # 标准差
            xn = xc / std  #

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class EmbeddingLayer(ILayer):
    _index: np.ndarray
    _weights: np.ndarray
    _weight_gradients: np.ndarray
    """
    词嵌入层，避免用one-hot变量（而是索引）来运算
    """

    def __init__(self, W: np.ndarray):
        self._weights = W
        self._index = None
        self._weight_gradients = np.zeros_like(W)

    def forward(self, index: np.ndarray):
        self._index = index
        return self._weights[index]

    def backward(self, dout):
        weight_gradients = self._weight_gradients
        # 每次反向传播时都要清空上次的，避免累加
        weight_gradients[...] = 0

        # 根据索引，将dout对应行累加到weight_gradients中（重复索引自动累加）
        np.add.at(weight_gradients, self._index, dout)

    def get_weights_gradients(self):
        return ([self._weights], [self._weight_gradients])


class EmbeddingDotLayer(ILayer):
    _embed_layer: EmbeddingLayer
    _in_vec: np.ndarray
    _out_vec: np.ndarray

    def __init__(self, W):
        self._embed_layer = EmbeddingLayer(W)

    def forward(self, in_vec: np.ndarray, idx: np.ndarray):
        self._in_vec = in_vec
        self._out_vec = out_vec = self._embed_layer.forward(idx)
        # axis=1 -> 按行求和，target * h是矩阵各位置元素相乘，而不是内积
        out = np.sum(out_vec * in_vec, axis=1)
        return out

    def backward(self, dout: np.ndarray):
        # 这里完全不用reshape（隐式（B，) * (B,D)会隐式变成(B,1) ）
        dout = dout.reshape(dout.shape[0], 1)
        d_out = dout * self._in_vec
        # 方向传播，以便后续更新W_out
        self._embed_layer.backward(d_out)
        d_in = dout * self._out_vec
        # 传到输入的embedding层，更新W_in
        return d_in

    def get_weights_gradients(self):
        return self._embed_layer.get_weights_gradients()


class NegativeSamplingLossLayer(ILayer):
    _negative_sampler: ISampler
    _loss_layers: list[SigmoidLossLayer]
    _embed_dot_layers: list[EmbeddingDotLayer]

    def __init__(self, vocab: Vocab, out_matrix: np.ndarray, sample_size=5, power=0.75):
        self._negative_sampler = SimpleSampler(
            vocab=vocab, sample_size=sample_size, power=power
        )
        loss_layers = self._loss_layers = []
        embed_dot_layers = self._embed_dot_layers = []
        for _ in range(sample_size + 1):
            loss_layers.append(SigmoidLossLayer())
            embed_dot_layer = EmbeddingDotLayer(out_matrix)
            embed_dot_layers.append(embed_dot_layer)

    def forward(self, in_vec: np.ndarray, target: np.ndarray):
        batch_size = target.shape[0]
        negative_smaples = self._negative_sampler.do_sample(target)
        embed_dot_layers = self._embed_dot_layers
        loss_layers = self._loss_layers
        # in_vec代表上下文词的综合向量，target代表中心词向量，两者内积就是得分
        score = embed_dot_layers[0].forward(in_vec=in_vec, idx=target)
        # 正例平均损失
        loss = loss_layers[0].forward(score, np.ones(batch_size, dtype=np.uint8))

        # 因为负样本和正样本息息相关，所以不能把正样本、负样本混在一个batch里（因为一个batch包含了多个样本），所以分多个loss_layer计算
        negative_labels = np.zeros(batch_size, dtype=np.uint8)
        for index in range(self._negative_sampler._sample_size):
            negative_target = negative_smaples[:, index]
            score = embed_dot_layers[index + 1].forward(
                in_vec=in_vec, idx=negative_target
            )
            loss += loss_layers[index + 1].forward(score, negative_labels)
        return loss

    def backward(self, dout):
        dh = 0
        for l0, l1 in zip(self._loss_layers, self._embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

    def get_weights_gradients(self):
        weights = []
        gradients = []
        for layer in self._embed_dot_layers:
            l_weights, l_gradients = layer.get_weights_gradients()
            weights.extend(l_weights)
            gradients.extend(l_gradients)
        return (weights, gradients)


class RNNLayer(ILayer):
    # h = tanh(x*wx + h_prev*wh + b)
    def __init__(self, wx, wh, b):

        pass

    def forward(self, x, h_prev):
        pass

    def backward(self, dout):
        pass
