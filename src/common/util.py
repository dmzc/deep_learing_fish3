import numpy as np
from src.common.vocab import Vocab
from src.common.logger import getLogger


def cos_similarity(x: np.ndarray, y: np.ndarray, eps=1e-8):
    """
    计算向量的余弦距离，判断相似度
    """
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def get_similar_words(query_word: str, vocab: Vocab, top=5):
    """
    查找相似词
    @param 查询词
    @param 词表
    """
    matrix = vocab.get_current_matrix()
    query_vector = vocab.get_word_vector(query_word)
    if query_vector is None:
        print(f"词表不包含查询字符串{query_word}")
        return None
    # @为矩阵每行与向量点积
    similarities: np.ndarray = (
        matrix
        @ query_vector
        / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector))
    )
    logger = getLogger()
    logger.info(f"[{query_word}]:")
    # argsort从小到大返回排序后索引数组，[::-1]反转排序数组，[:top]取前n个
    sorded_similarity_word_ids = similarities.argsort()[::-1]
    count = 0
    words: list[str] = []
    for index in range(0, len(sorded_similarity_word_ids) - 1):
        word_id = sorded_similarity_word_ids[index]
        word = vocab.get_word(word_id)
        if not word == query_word:
            words.append(word)
            logger.info(f"{word}:{similarities[word_id]}")
            count += 1
        if count >= top:
            break

    return words


def cross_entropy(y, t: np.ndarray):
    """
    交叉熵损失函数
    """
    if y.ndim == 1:
        # 将单样本统一为多样本形式
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引

    # axis为1时按行比较，为0时按列比较

    # t = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  => [1, 2, 3]
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]  # 样本数

    # np.arange(batch_size)生成一个样本数相同的数组，代表取第几个样本的值，t为多个样本最大值数组索引。

    # 二维数组的索引取数组，代表要批量操作
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax(x: np.typing.NDArray[np.number]):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
