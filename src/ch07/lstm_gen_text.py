from pathlib import Path
import pickle
from src.common.models import LSTMModel, LSTMModelParams
import numpy as np


src_path = Path(__file__).parent.parent
pkl_path = src_path / "ch06" / "BetterRnnlm_n.pkl"

if not Path.exists(pkl_path):
    raise IOError("权重文件不存在: " + pkl_path.as_uri())
model: LSTMModel
with open(pkl_path, "rb") as f:
    lstm_model_params: LSTMModelParams = pickle.load(f)
    model = LSTMModel(lstm_model_params)
    start_id = model.get_id_by_word("you")
    skip_ids = [model.get_id_by_word(w) for w in ["N", "<unk>", "$"]]
    ids = model.generate(start_word=start_id, skip_words=skip_ids, sample_size=10)

    txt = " ".join([model.get_word_by_id(id) for id in ids])
    print(txt)

    model.reset_state()
    start_ids = [
        model.get_id_by_word(word) for word in "the meaning of life is".split(" ")
    ]
    # 模型预热，将前n-1个填到状态里
    for x in start_ids[:-1]:
        x = np.array(x).reshape(1, 1)
        model.predict(x)

    ids = model.generate(start_word=start_ids[-1], skip_words=skip_ids, sample_size=5)
    ids = start_ids[:-1] + ids
    txt = " ".join([model.get_word_by_id(id) for id in ids])
    print("-" * 50)
    print(txt)
