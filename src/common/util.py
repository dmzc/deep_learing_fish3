import numpy as np
from src.common.vocab import Vocab
from src.common.logger import getLogger
import sys
import os


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


def cross_entropy(y: np.ndarray, t: np.ndarray):
    """
    交叉熵损失函数
    """
    if y.ndim == 1:
        # 将单样本统一为多样本形式，统一后续操作
        # y = [0.1, 0.2 ,0.7], t = [0, 1, 0]
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        # axis为1时按行比较，为0时按列比较
        # t = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  => [1, 2, 3]
        t = t.argmax(axis=1)

    batch_size = y.shape[0]  # 样本数

    # np.arange(batch_size)生成一个样本数相同的数组，代表取第几个样本的值，t为多个样本最大值数组索引。
    # 二维数组的索引取数组，代表要批量操作
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax(x: np.typing.NDArray[np.number]) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def clip_grads(grads: np.ndarray, max_grad: int):
    if max_grad is None:
        return
    total_grad = 0
    for grad in grads:
        total_grad += np.sum(grad**2)
    total_grad = np.sqrt(total_grad)

    rate = max_grad / (total_grad + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def eval_perplexity(model, vocab: Vocab, batch_size=10, time_size=35):
    print("evaluating perplexity ...")
    corpus_size = vocab.get_corpus_size()
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                pass
                # xs[i, t] = corpus[(offset + t) % corpus_size]
                # ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write("\r%d / %d" % (iters, max_iters))
        sys.stdout.flush()

    print("")
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(
    model,
    question: np.ndarray,
    answer: np.ndarray,
    id_to_char: dict[int, str],
    char_to_id: dict[str, int],
    verbose=True,
) -> int:
    answer = answer.flatten()
    start_id = answer[0]
    answer = answer[0:]
    guess = model.generate(xs=question, start_id=start_id, sample_size=len(answer))

    # 转换为字符串
    question = "".join([id_to_char[int(c)] for c in question.flatten()])
    answer = "".join([id_to_char[int(c)] for c in answer])
    guess = "".join([id_to_char[int(c)] for c in guess])
    if verbose:
        colors = {"ok": "\033[92m", "fail": "\033[91m", "close": "\033[0m"}
        print("Q", question)
        print("T", answer)

        is_windows = os.name == "nt"

        if answer == guess:
            mark = colors["ok"] + "☑" + colors["close"]
            if is_windows:
                mark = "O"
            print(mark + " " + guess)
        else:
            mark = colors["fail"] + "☒" + colors["close"]
            if is_windows:
                mark = "X"
            print(mark + " " + guess)
        print("---")

    return 1 if guess == answer else 0
