import numpy as np
from src.common.interfaces.IVocab import IVocab
from src.common.decorators.timer import timer
import time as time


class Vocab(IVocab):
    __word_id: dict[str, int]
    __id_word: dict[int, str]
    __corpus: np.ndarray
    __co_matrixs: dict[int, np.ndarray]
    __ppmi_matrixs: dict[int, np.ndarray]
    __ppmi_svd_matrixs: dict[int, np.ndarray]

    __current_window_size: int
    __current_use_PPMI: bool
    __current_use_SVD: bool
    __current_svd_vector_size: int

    __probability: np.ndarray

    def __init__(self):
        self.__word_id = {}
        self.__id_word = {}
        self.__corpus = np.zeros(0)
        self.__co_matrixs = {}
        self.__ppmi_matrixs = {}
        self.__ppmi_svd_matrixs = {}
        self.__current_window_size = 2
        self.__current_use_PPMI = False
        self.__current_use_SVD = False
        self.__current_svd_vector_size = 100
        self.__probability = None

    def build(self, text: str) -> np.ndarray:
        # TODO:这里不能在拼接而成的大句子里构建共现矩阵，而是应该在每个句子内滑动
        wordsArr = text.lower().replace(".", " .").split(" ")
        wordsMap: dict[str, int] = {}
        idsMap: dict[int, str] = {}
        for word in wordsArr:
            if word not in wordsMap:
                id = len(wordsMap)
                wordsMap[word] = id
                idsMap[id] = word
        corpus = np.array([wordsMap[word] for word in wordsArr], dtype=np.uint32)
        self.__word_id = wordsMap
        self.__id_word = idsMap
        self.__corpus = corpus
        self.__co_matrixs = {}
        return corpus

    def get_word(self, id: int) -> str:

        if id in self.__id_word:
            return self.__id_word[id]
        return None

    def get_words(self) -> list[str]:
        return self.__id_word

    def get_id(self, word: str) -> int:
        if word in self.__word_id:
            return self.__word_id[word]
        return None

    def has_word(self, word: str) -> bool:
        return word in self.__word_id

    def get_word_vector(self, word: str) -> np.ndarray:
        if not self.has_word(word):
            return None
        return self.get_current_matrix()[self.get_id(word)]

    def get_matirx(
        self, window_size=2, use_PPMI=False, use_SVD=False, svd_vector_size=100
    ) -> np.ndarray:
        if use_PPMI:
            return self._get_ppmi_matrix(window_size)
        if use_SVD:
            return self._get_ppmi_matrix(window_size, True, svd_vector_size)
        return self._get_co_matrix(window_size)

    def set_current_matrix(
        self,
        window_size: int = 2,
        use_PPMI: bool = False,
        use_SVD: bool = False,
        svd_vector_size: int = 100,
    ):
        self.__current_window_size = window_size
        self.__current_svd_vector_size = svd_vector_size
        self.__current_use_PPMI = use_PPMI
        self.__current_use_SVD = use_SVD

    def get_current_matrix(self) -> np.ndarray:
        return self.get_matirx(
            self.__current_window_size,
            self.__current_use_PPMI,
            self.__current_use_SVD,
            self.__current_svd_vector_size,
        )

    def get_size(self) -> int:
        return len(self.__id_word)

    def get_corpus_size(self) -> int:
        return len(self.__corpus)

    def get_current_window_size(self) -> int:
        return self.__current_window_size

    @timer(message="获取word2vec数据耗时")
    def get_word2vec_data(self, window_size=2, use_id=True, use_onehot=False):
        """
        获取word2vec的训练数据
        return (context_words, center_words)
        """
        corpus = self.__corpus
        corpus_size = len(corpus)

        # 中心词，中心词索引
        # corpus = [10, 20, 30, 40, 50]，window_size = 1
        # center_words = [20, 30, 40]，center_words_idx = [1, 2, 3]
        # 注意：arange是左开右闭，所以是-window_size
        center_words_id = corpus[window_size : corpus_size - window_size]
        center_words_idx = np.arange(window_size, corpus_size - window_size)
        # 将(3,)转换为(3,1)，既[ [1], [2], [3] ]
        center_words_idx = center_words_idx[:, None]
        # 生成上下文词偏移量，[-1, 1]
        offsets = np.hstack([np.arange(-window_size, 0), np.arange(1, window_size + 1)])

        # center_words_idx + offsets，其中center_words_idx为(3,1)，会先将offsets(2)扩充为(1,2)
        # 相加时自动广播，会扩展成3 * 2的矩阵，相加只有形状相关才能进行，但是如果有1，则可以扩展
        # center_words_idx + offsets就变成了3 * 2的矩阵，[ [0, 2], [1, 3], [2, 4] ]
        context_words_idx = center_words_idx + offsets

        # corpus[contexts_idx]操作则是按矩阵中的idx作为索引去取corpus中元素
        context_words_id = corpus[context_words_idx]

        if use_id:
            if use_onehot:
                vocab_size = len(self.__id_word)
                identity_matrix = np.eye(vocab_size, dtype=np.uint8)
                return identity_matrix[context_words_id], identity_matrix[
                    center_words_id
                ]
            return context_words_id, center_words_id
        word_vectorize = np.vectorize(self.__id_word.get)
        return word_vectorize(context_words_id), word_vectorize(center_words_id)

    def _get_co_matrix(self, window_size=2) -> np.ndarray:
        """
        获取词表的共现矩阵
        """
        __co_matrixs = self.__co_matrixs
        if window_size not in __co_matrixs:
            co_matrix = self._get_co_matrix_vector(window_size)
            __co_matrixs[window_size] = co_matrix
        return __co_matrixs[window_size]

    def _get_ppmi_matrix(self, window_size=2, SVD=False, vector_size=100) -> np.ndarray:
        """
        获取词表的正点互信息矩阵
        """
        __ppmi_matrixs = self.__ppmi_matrixs
        __ppmi_svd_matrixs = self.__ppmi_svd_matrixs

        def do_svd(ppmi_matrix: np.ndarray):
            __ppmi_svd_matrix = self._ppmi_matrix_svd(ppmi_matrix, vector_size)
            __ppmi_svd_matrixs[window_size] = __ppmi_svd_matrix
            return __ppmi_svd_matrix

        if window_size in __ppmi_matrixs:
            ppmi_matrix = __ppmi_matrixs[window_size]
            if SVD:
                if window_size in __ppmi_svd_matrixs:
                    return __ppmi_svd_matrixs[window_size]
                else:
                    return do_svd(ppmi_matrix)
            else:
                return ppmi_matrix
        ppmi_matrix = self._get_ppmi_matrix_vector(window_size)
        __ppmi_matrixs[window_size] = ppmi_matrix

        if SVD:
            return do_svd(ppmi_matrix)
        else:
            return ppmi_matrix

    @timer(message="使用原始for循环构建共现矩阵")
    def _get_co_matrix_raw(self, window_size=2):
        corpus = self.__corpus
        corpus_size = len(corpus)
        vocab_size = len(self.__id_word)
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for index, word_id in enumerate(corpus):
            # range是左开右闭区间，range(1,2)->1
            for offset in range(1, window_size + 1):
                left_index = index - offset
                right_index = index + offset
                if left_index >= 0:
                    left_id = corpus[left_index]
                    co_matrix[word_id, left_id] += 1
                if right_index < corpus_size:
                    right_id = corpus[right_index]
                    co_matrix[word_id, right_id] += 1
        return co_matrix

    @timer(message="使用向量化方式构建共现矩阵")
    def _get_co_matrix_vector(self, window_size=2) -> np.ndarray:
        # TODO:这里向量化方式计算待搞懂
        corpus = self.__corpus
        corpus_size = len(corpus)
        vocab_size = len(self.__id_word)
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        # 向量形式构建共现矩阵
        half_window = window_size  # 原代码的有效窗口半宽

        # 1. 构造所有窗口偏移量
        offsets = np.concatenate(
            [
                np.arange(-half_window, 0),  # 左边偏移: -1, -2, ..., -half_window
                np.arange(1, half_window + 1),  # 右边偏移: 1, 2, ..., half_window
            ]
        )

        # 2. 构造每个词对应的邻居索引（超出边界的部分用-1标记）
        indices = (
            np.arange(corpus_size)[:, None] + offsets
        )  # shape: (corpus_size, 2*(window_size-1))

        # 3. 过滤掉超出边界的索引
        valid_mask = (indices >= 0) & (indices < corpus_size)
        valid_indices = indices[valid_mask]  # 只保留有效的邻居索引

        # 4. 获取中心词id和对应的邻居词id
        center_ids = np.repeat(corpus, valid_mask.sum(axis=1))
        neighbor_ids = corpus[valid_indices]

        # 5. 一次性更新共现矩阵
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
        np.add.at(co_matrix, (center_ids, neighbor_ids), 1)
        return co_matrix

    @timer(message="使用原始for循环构建PPMI矩阵")
    def _get_ppmi_matrix_raw(self, window_size=2) -> np.ndarray:
        co_matrix = self._get_co_matrix(window_size)
        ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float64)
        all_count = np.sum(co_matrix)  # 总词对次数
        # 行和(汇总列)：词作为中心词时出现的总次数（i的总次数）
        target_word_count = np.sum(co_matrix, axis=1)
        # 列和(汇总行)：词作为上下文词时出现的总次数（j的总次数）
        context_word_count = np.sum(co_matrix, axis=0)
        eps = 1e-8
        for i in range(co_matrix.shape[0]):
            for j in range(co_matrix.shape[1]):
                # x为中心词，y为上下文词。
                # p(x) = row_sum[i]/N
                # p(y) = col_sum[j]/N
                # p(x,y) = C[i,j]
                pmi = np.log2(
                    (co_matrix[i, j] * all_count + eps)
                    / (target_word_count[i] * context_word_count[j] + eps)
                )
                ppmi_matrix[i, j] = max(0, pmi)  # PPMI：取max(0, pmi)
        return ppmi_matrix

    @timer(message="使用向量化方式构建PPMI矩阵")
    def _get_ppmi_matrix_vector(self, window_size=2) -> np.ndarray:
        co_matrix = self._get_co_matrix(window_size)
        ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float64)
        all_count = np.sum(co_matrix)  # 总词对次数
        # 行和(汇总列)：词作为中心词时出现的总次数（i的总次数）
        target_word_count = np.sum(co_matrix, axis=1)
        # 列和(汇总行)：词作为上下文词时出现的总次数（j的总次数）
        context_word_count = np.sum(co_matrix, axis=0)
        eps = 1e-8

        # 1. 向量化计算 PPMI，完全去掉双重循环
        # 计算分子
        numerator = co_matrix * all_count + eps
        # 计算分母，target_word_conut的形状为(V，)，转换为(V,1),context_word_count的形状为(V,)，转换为(1,V)
        denominator = (
            target_word_count[:, np.newaxis] * context_word_count[np.newaxis, :] + eps
        )
        pmi_matrix = np.log2(numerator / denominator)
        ppmi_matrix = np.maximum(0, pmi_matrix)  # PPMI: 取max(0, pmi)
        return ppmi_matrix

    @timer(message="PPMI矩阵使用svd降维耗时")
    def _ppmi_matrix_svd(
        self, ppmi_matrix: np.ndarray, vector_size=100, use_acceleration=True
    ) -> np.ndarray:
        if use_acceleration:
            from sklearn.utils.extmath import randomized_svd

            U, S, V = randomized_svd(
                ppmi_matrix,
                n_components=vector_size,
                n_iter=5,
                random_state=None,
            )
        else:
            # 9274 * 9274的矩阵，使用svd降维需要266秒，使用sklearn中加速只需要3.48秒
            U, s, VT = np.linalg.svd(ppmi_matrix, full_matrices=False)
            U = U[:, :vector_size]  # 这就是最终词向量
        return U

    def get_probability(self) -> np.ndarray:
        if self.__probability is not None:
            return self.__probability
        corpus = self.__corpus
        word_count = np.bincount(corpus, minlength=self.get_size())
        self.__probability = word_count / len(corpus)
        return self.__probability

    def get_rnn_data(self) -> np.ndarray:

        # self.__corpus取除最后一个词之外的所有词。self.__corpus[1:]取除第一个词之外的所有词
        # 因为rnn的任务是预测下一个词，所以这么处理，按相同index去取时，t永远是x的下一个词
        return self.__corpus[:-1], self.__corpus[1:]
