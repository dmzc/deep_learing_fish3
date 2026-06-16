from __future__ import annotations
import numpy as np
from abc import ABC
from dataclasses import dataclass


class IModel(ABC):
    # 所有层的权重
    _weights: list[np.ndarray]

    # 所有层的梯度
    _gradients: list[np.ndarray]

    def get_weights_gradients() -> tuple[list[np.ndarray], list[np.ndarray]]: ...

    def forward(self, contexts, target) -> float: ...

    def backward(self, dout) -> None: ...


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
