from __future__ import annotations
import numpy as np
from abc import ABC
from dataclasses import dataclass
from src.common.layers import (
    ILayer,
    AffineLayer,
    SoftmaxLossLayer,
    EmbeddingLayer,
    NegativeSamplingLossLayer,
    TimeEmbeddingLayer,
    TimeRNNLayer,
    TimeAffineLayer,
    TimeSoftmaxLossLayer,
    TimeLSTMLayer,
    TimeDropoutLayer,
    TimeEncoderLayer,
    TimeDecoderLayer,
    TimePeekyDecoderLayer,
)
from src.common.vocab import Vocab
from src.common.util import softmax


class IModel(ABC):
    # 所有层的权重
    _weights: list[np.ndarray]

    # 所有层的梯度
    _gradients: list[np.ndarray]

    def get_weights_gradients() -> tuple[list[np.ndarray], list[np.ndarray]]: ...

    def forward(self, contexts, target) -> float: ...

    def backward(self, dout) -> None: ...


class AbstractModel(IModel):
    def get_weights_gradients(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        layers = self.get_weight_layers()
        if layers is None:
            return None

        __weightsMap = {}
        __gradientsMap = {}

        def __collect(layer: ILayer):
            weights_grdients: tuple[list[np.ndarray], list[np.ndarray]] = (
                layer.get_weights_gradients()
            )
            if weights_grdients is not None:
                weights = weights_grdients[0]
                gradients = weights_grdients[1]
                for index, weight in enumerate(weights):
                    key = id(weight)
                    gradient = gradients[index]
                    if key in __weightsMap:
                        t_gradient = __gradientsMap[key]
                        t_gradient += gradient
                    else:
                        # 严重失误，weight怎么能copy呢？copy之后不就无法按梯度更新了吗？
                        __weightsMap[key] = weight
                        __gradientsMap[key] = gradient.copy()
            sub_layers = layer.get_sub_weight_layers()
            if sub_layers is not None:
                for s_layer in sub_layers:
                    __collect(s_layer)

        for layer in layers:
            __collect(layer)
        weights = []
        gradients = []
        for key in __weightsMap:
            weights.append(__weightsMap.get(key))
            gradients.append(__gradientsMap.get(key))
        return (weights, gradients)

    def get_weight_layers(self) -> list[ILayer]:
        pass


class SimpleCbowModel(AbstractModel):
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

    def get_weight_layers(self):
        return [self.__in_layer1, self.__in_layer2, self.__out_layer]

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


class CbowModel(AbstractModel):
    # 输入层矩阵
    _W_in: np.ndarray
    # 输出层矩阵
    _W_out: np.ndarray

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
        for _ in range(window_size * 2):
            in_layer = EmbeddingLayer(W_in)
            in_layers.append(in_layer)

        self._loss_layer = NegativeSamplingLossLayer(vocab=vocab, out_matrix=W_out)

    def get_weight_layers(self):
        layers = []
        layers.extend(self._in_layers)
        layers.append(self._loss_layer)
        return layers

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


class RNNModel(AbstractModel):
    __layers: list[ILayer]
    __loss_layer: ILayer

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size):

        embed_w = (np.random.randn(vocab_size, wordvec_size) / 100).astype(np.float32)
        rnn_wx: np.ndarray = np.random.randn(wordvec_size, hidden_size) / np.sqrt(
            wordvec_size
        )
        rnn_wx = rnn_wx.astype(np.float32)
        rnn_wh: np.ndarray = np.random.randn(hidden_size, hidden_size) / np.sqrt(
            wordvec_size
        )
        rnn_wh = rnn_wh.astype(np.float32)
        rnn_b = np.zeros(hidden_size).astype(np.float32)
        affine_w: np.ndarray = np.random.randn(hidden_size, vocab_size) / np.sqrt(
            hidden_size
        )
        affine_w = affine_w.astype(np.float32)
        affine_b = np.zeros(vocab_size).astype(np.float32)

        self.__layers = [
            TimeEmbeddingLayer(embed_w),
            TimeRNNLayer(wx=rnn_wx, wh=rnn_wh, b=rnn_b),
            TimeAffineLayer(wx=affine_w, wb=affine_b),
        ]
        self.__loss_layer = TimeSoftmaxLossLayer()

    def forward(self, xs: np.ndarray, ts: np.ndarray):
        for layer in self.__layers:
            xs = layer.forward(xs)
        return self.__loss_layer.forward(xs, ts)

    def backward(self, dout=1):
        dout = self.__loss_layer.backward(dout)
        for layer in reversed(self.__layers):
            dout = layer.backward(dout)
        return dout

    def get_weight_layers(self):
        return self.__layers


@dataclass
class LSTMModelParams:
    vocab_size: int
    wordvec_size: int
    hiddent_size: int
    words: dict[int, str]
    dropout_ratio: float = 0.5

    # 下面参数为权重，要不同时为None，要不同时不为空。不为空时代表是推理阶段要加载参数
    embedding1_wx: np.ndarray = None
    lstm1_wx: np.ndarray = None
    lstm1_wh: np.ndarray = None
    lstm1_wb: np.ndarray = None
    lstm2_wx: np.ndarray = None
    lstm2_wh: np.ndarray = None
    lstm2_wb: np.ndarray = None
    affine1_wb: np.ndarray = None


class LSTMModel(AbstractModel):
    __layers: list[ILayer]

    __dropout_layers: list[TimeDropoutLayer]

    __lstm_layers: list[TimeLSTMLayer]

    __loss_layer: ILayer

    __id_word: dict[int, str]

    __word_id: dict[str, int]

    def __init__(self, params: LSTMModelParams):
        rn = np.random.randn
        temp: np.ndarray

        self.__id_word = params.words
        word_id = self.__word_id = {}
        for key in self.__id_word:
            word_id[self.__id_word.get(key)] = key

        vocab_size: int = params.vocab_size
        word_vec_size: int = params.wordvec_size
        hidden_size: int = params.hiddent_size
        dropout_ratio: float = params.dropout_ratio

        embedding1_wx = params.embedding1_wx
        lstm1_wx = params.lstm1_wx
        lstm1_wh = params.lstm1_wh
        lstm1_wb = params.lstm1_wb
        lstm2_wx = params.lstm2_wx
        lstm2_wh = params.lstm2_wh
        lstm2_wb = params.lstm2_wb
        affine1_b = params.affine1_wb

        if params.embedding1_wx is None:  # 非加载模式
            temp = rn(vocab_size, word_vec_size) / 100
            embedding1_wx = temp.astype(np.float32)
            temp = rn(word_vec_size, 4 * hidden_size) / np.sqrt(word_vec_size)
            lstm1_wx = temp.astype(np.float32)
            temp = rn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
            lstm1_wh = temp.astype(np.float32)
            lstm1_wb = np.zeros(4 * hidden_size).astype(np.float32)
            temp = rn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
            lstm2_wx = temp.astype(np.float32)
            temp = rn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
            lstm2_wh = temp.astype(np.float32)
            lstm2_wb = np.zeros(4 * hidden_size).astype(np.float32)
            affine1_b = np.zeros(vocab_size).astype(np.float32)

        self.__layers = [
            TimeEmbeddingLayer(embedding1_wx),
            TimeDropoutLayer(dropout_ratio=dropout_ratio),
            TimeLSTMLayer(wx=lstm1_wx, wh=lstm1_wh, wb=lstm1_wb, stateful=True),
            TimeDropoutLayer(dropout_ratio=dropout_ratio),
            TimeLSTMLayer(wx=lstm2_wx, wh=lstm2_wh, wb=lstm2_wb, stateful=True),
            TimeDropoutLayer(dropout_ratio=dropout_ratio),
            TimeAffineLayer(embedding1_wx, affine1_b, weight_typing=True),
        ]
        self.__loss_layer = TimeSoftmaxLossLayer()
        self.__dropout_layers = [self.__layers[1], self.__layers[3], self.__layers[5]]
        self.__lstm_layers = [self.__layers[2], self.__layers[4]]

    def predict(self, xs):
        for layer in self.__dropout_layers:
            layer.set_training_state(False)
        score = self._forward_no_loss(xs)
        for layer in self.__dropout_layers:
            layer.set_training_state(True)
        return score

    def forward(self, xs, ts):
        score = self._forward_no_loss(xs)
        return self.__loss_layer.forward(score, ts)

    def backward(self, dout=1):
        dout = self.__loss_layer.backward(dout)
        for layer in reversed(self.__layers):
            dout = layer.backward(dout)
        return dout

    def _forward_no_loss(self, xs):
        for layer in self.__layers:
            xs = layer.forward(xs)
        return xs

    def generate(self, start_word: int, skip_words: list[int] = None, sample_size=100):
        words = [start_word]
        x: np.ndarray = None
        while len(words) < sample_size:
            if x is None:
                x = start_word
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if skip_words is None or sampled not in skip_words:
                x = sampled
                words.append(int(x))
        return words

    def get_weight_layers(self):
        return self.__layers

    def reset_state(self):
        for layer in self.__lstm_layers:
            layer.reset_state()

    def get_id_by_word(self, word: str) -> int:
        return self.__word_id.get(word)

    def get_word_by_id(self, id: int) -> str:
        return self.__id_word.get(id)


@dataclass
class Seq2SeqModelParams:
    # TODO:支持加载参数
    vocab_size: int
    wordvec_size: int
    hiddent_size: int
    words: dict[int, str]

    # 下面参数为权重，要不同时为None，要不同时不为空。不为空时代表是推理阶段要加载参数
    embedding1_wx: np.ndarray = None
    lstm1_wx: np.ndarray = None
    lstm1_wh: np.ndarray = None
    lstm1_wb: np.ndarray = None
    lstm2_wx: np.ndarray = None
    lstm2_wh: np.ndarray = None
    lstm2_wb: np.ndarray = None
    affine1_wb: np.ndarray = None


class Seq2SeqModel(AbstractModel):
    __encoder_layer: TimeEncoderLayer
    __decoder_layer: TimeDecoderLayer
    __loss_layer: TimeSoftmaxLossLayer

    def __init__(
        self, vocab_size: int, wordvec_size: int, hidden_size: int, use_peeky=False
    ):
        self.__encoder_layer = TimeEncoderLayer(vocab_size, wordvec_size, hidden_size)
        if use_peeky:
            self.__decoder_layer = TimePeekyDecoderLayer(
                vocab_size, wordvec_size, hidden_size
            )
        else:
            self.__decoder_layer = TimeDecoderLayer(
                vocab_size, wordvec_size, hidden_size
            )

        self.__loss_layer = TimeSoftmaxLossLayer()

    def forward(self, xs: np.ndarray, ts: np.ndarray):
        score = self.__decoder_layer.set_state(
            h=self.__encoder_layer.forward(xs)
        ).forward(xs=ts[:, :-1])
        loss = self.__loss_layer.forward(score, ts[:, 1:])
        return loss

    def backward(self, dout=1):
        dout = self.__loss_layer.backward(dout)
        dh = self.__decoder_layer.backward(dout)
        dout = self.__encoder_layer.backward(dh)
        return dout

    def generate(self, xs: np.ndarray, start_id: int, sample_size: int):
        return self.__decoder_layer.set_state(
            h=self.__encoder_layer.forward(xs)
        ).generate(start_id=start_id, sample_size=sample_size)

    def get_weight_layers(self):
        return [self.__decoder_layer, self.__encoder_layer]
