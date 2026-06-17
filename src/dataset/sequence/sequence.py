from pathlib import Path
import numpy as np

current_folder = Path(__file__).parent

id_to_char = {}
char_to_id = {}


def __update_vocab(text: str) -> None:
    for index, v_char in enumerate(list(text)):
        if v_char not in char_to_id:
            id = len(id_to_char)
            id_to_char[id] = v_char
            char_to_id[v_char] = id


def load_data(file="addition.txt", seed=1984) -> list[tuple[np.ndarray, np.ndarray]]:
    file_path = current_folder / file
    questions, answers = [], []
    with open(file_path, "r") as text_iter:
        for line in text_iter:
            idx = line.find("_")
            question = line[:idx]
            answer = line[idx + 1 : -1]
            questions.append(question)
            answers.append(answer)
            __update_vocab(question)
            __update_vocab(answer)

    data_length = len(questions)
    x = np.zeros((data_length, len(questions[0])), dtype=np.int32)
    t = np.zeros((data_length, len(answers[0])), dtype=np.int32)

    for index in range(data_length):
        question = questions[index]
        answer = answers[index]
        x[index] = [char_to_id[v_char] for v_char in list(question)]
        t[index] = [char_to_id[v_char] for v_char in list(answer)]

    # 打乱数据
    indices = np.arange(data_length)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    split_at = data_length - data_length // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return [(x_train, t_train), (x_test, t_test)]


def get_vocab() -> tuple[dict[str, int], dict[int, str]]:
    return char_to_id, id_to_char
