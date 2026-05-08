import numpy as np
import time
from src.common.logger import getLogger


class Vocab:
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

    def build(self, text: str) -> np.ndarray:
        wordsArr = text.lower().replace(".", " .").split(" ")
        wordsMap: dict[str, int] = {}
        idsMap: dict[int, str] = {}
        start_time = time.time()
        for word in wordsArr:
            if word not in wordsMap:
                id = len(wordsMap)
                wordsMap[word] = id
                idsMap[id] = word
        end_time = time.time()
        getLogger().info(f"执行耗时{end_time - start_time}")
        corpus = np.array([wordsMap[word] for word in wordsArr])
        self.__word_id = wordsMap
        self.__id_word = idsMap
        self.__corpus = corpus
        self.__co_matrixs = {}
        return corpus

    def get_word(self, id: int) -> str:

        if id in self.__id_word:
            return self.__id_word[id]
        return None

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

    def _get_co_matrix(self, window_size=2) -> np.ndarray:
        """
        获取词表的共现矩阵
        """
        __co_matrixs = self.__co_matrixs
        if window_size not in __co_matrixs:
            # 根据语料创建单词的共现矩阵
            corpus = self.__corpus
            vocab_size = len(self.__id_word)
            corpus_size = len(corpus)
            co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
            for index, word_id in enumerate(corpus):
                for offset in range(1, window_size + 1):
                    left_index = index - offset
                    right_index = index + offset
                    if left_index >= 0:
                        left_id = corpus[left_index]
                        co_matrix[word_id, left_id] += 1
                    if right_index < corpus_size:
                        right_id = corpus[right_index]
                        co_matrix[word_id, right_id] += 1
            __co_matrixs[window_size] = co_matrix
        return __co_matrixs[window_size]

    def _get_ppmi_matrix(self, window_size=2, SVD=False, vector_size=3) -> np.ndarray:
        """
        获取词表的正点互信息矩阵
        """
        __ppmi_matrixs = self.__ppmi_matrixs
        __ppmi_svd_matrixs = self.__ppmi_svd_matrixs

        if window_size in __ppmi_matrixs:
            if SVD:
                if window_size in __ppmi_svd_matrixs:
                    return __ppmi_svd_matrixs[window_size]
                else:
                    self._do_SVD(window_size, vector_size)
                    return __ppmi_svd_matrixs[window_size]
            else:
                return __ppmi_matrixs[window_size]

        start_time = time.time()
        co_matrix = self._get_co_matrix(window_size)
        end_time = time.time()
        getLogger().info(f"构建共现矩阵耗时{end_time - start_time}")

        ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float64)

        all_count = np.sum(co_matrix)  # 总词对次数
        # 行和(汇总列)：词作为中心词时出现的总次数（i的总次数）
        target_word_count = np.sum(co_matrix, axis=1)
        # 列和(汇总行)：词作为上下文词时出现的总次数（j的总次数）
        context_word_count = np.sum(co_matrix, axis=0)
        eps = 1e-8
        # start_time = time.time()
        # for i in range(co_matrix.shape[0]):
        #     for j in range(co_matrix.shape[1]):
        #         # x为中心词，y为上下文词。
        #         # p(x) = row_sum[i]/N
        #         # p(y) = col_sum[j]/N
        #         # p(x,y) = C[i,j]
        #         pmi = np.log2(
        #             (co_matrix[i, j] * all_count + eps)
        #             / (target_word_count[i] * context_word_count[j] + eps)
        #         )
        #         ppmi_matrix[i, j] = max(0, pmi)  # PPMI：取max(0, pmi)
        # __ppmi_matrixs[window_size] = ppmi_matrix
        # end_time = time.time()
        # getLogger().info(f"for循环计算PPMI耗时: {end_time - start_time:.4f} 秒")

        start_time = time.time()
        # 1. 向量化计算 PPMI，完全去掉双重循环
        # 计算分子
        numerator = co_matrix * all_count + eps
        # 计算分母，target_word_conut的形状为(V，)，转换为(V,1),context_word_count的形状为(V,)，转换为(1,V)
        denominator = (
            target_word_count[:, np.newaxis] * context_word_count[np.newaxis, :] + eps
        )
        pmi_matrix = np.log2(numerator / denominator)
        ppmi_matrix = np.maximum(0, pmi_matrix)  # PPMI: 取max(0, pmi)
        __ppmi_matrixs[window_size] = ppmi_matrix

        end_time = time.time()
        getLogger().info(f"向量化计算PPMI耗时: {end_time - start_time:.4f} 秒")

        if SVD:
            start_time = time.time()
            self._do_SVD(window_size, vector_size)
            end_time = time.time()
            getLogger().info(f"共现矩阵降维耗时{end_time - start_time}")
            return __ppmi_svd_matrixs[window_size]
        else:
            return ppmi_matrix

    def get_matirx(
        self, window_size=2, use_PPMI=False, use_SVD=False, svd_vector_size=100
    ) -> np.ndarray:
        if use_PPMI:
            return self._get_co_matrix(window_size)
        if use_SVD:
            self._get_ppmi_matrix(window_size, True, svd_vector_size)
        return self._get_ppmi_matrix(window_size, False, svd_vector_size)

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

    def _do_SVD(self, window_size: int, vec_size: int) -> np.ndarray:
        from sklearn.utils.extmath import randomized_svd

        U, S, V = randomized_svd(
            self.__ppmi_matrixs[window_size],
            n_components=vec_size,
            n_iter=5,
            random_state=None,
        )
        self.__ppmi_svd_matrixs[window_size] = U
        pass
