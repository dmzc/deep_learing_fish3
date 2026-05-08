import numpy as np
from src.common.vocab import Vocab
from src.common.logger import getLogger
from src.common.enums.similar_strategy import SimilarStrategy


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
