from src.common.vocab import Vocab
from src.common.util import cos_similarity, get_similar_words
import numpy as np


def test_cos_similarity():

    vocab = Vocab()
    vocab.build("You say goodbye and I say hello.")
    i_vec = vocab.get_word_vector("i")
    you_vec = vocab.get_word_vector("you")
    goodbye_vec = vocab.get_word_vector("goodbye")
    hello_vec = vocab.get_word_vector("hello")
    assert np.allclose(cos_similarity(i_vec, i_vec), 1.0), "同一向量余弦值距离为1"
    assert round(float(cos_similarity(i_vec, you_vec)), 3) == 0.707, (
        "相似度高的词余弦值高"
    )
    assert round(float(cos_similarity(goodbye_vec, hello_vec)), 3) == 0.577, (
        "相似度低的词余弦值低"
    )


def test_most_similarity():
    vocab = Vocab()
    vocab.build("You say goodbye and I say hello.")

    assert get_similar_words("say", vocab, 2) == ["goodbye", "i"]
