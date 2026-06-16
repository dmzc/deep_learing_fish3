import os
import pickle
import numpy as np
from pathlib import Path
from src.common.models import LSTMModelParams, LSTMModel
from src.dataset.ptb import ptb
from src.common.interfaces.IVocab import IVocab

cur_path = Path(__file__).parent

pkl_path1 = cur_path / "Rnnlm.pkl"
n_pkl_path1 = cur_path / "Rnnlm_n.pkl"

pkl_path2 = cur_path / "BetterRnnlm.pkl"
n_pkl_path2 = cur_path / "BetterRnnlm_n.pkl"


print(ptb.load_data(use_cache=False).get_size())

if not os.path.exists(pkl_path1):
    raise IOError("No file: " + pkl_path1.as_uri())

with open(pkl_path1, "rb") as f:
    params = pickle.load(f)
    params_dict: dict[str, np.ndarray] = {}
    params_dict["vocab_size"] = 100000
    params_dict["wordvec_size"] = 100
    params_dict["hidden_size"] = 100
    params_dict["embedding1_w"] = params[0]
    params_dict["lstm1_wx"] = params[1]
    params_dict["lstm1_wh"] = params[2]
    params_dict["lstm1_wb"] = params[3]
    params_dict["affine1_wx"] = params[4]
    params_dict["affine1_wb"] = params[5]
with open(n_pkl_path1, "wb") as f:
    pickle.dump(params_dict, f)
    pass


if not os.path.exists(pkl_path2):
    raise IOError("No file: " + pkl_path2.as_uri())
params_dict = {}
with open(pkl_path2, "rb") as f:
    params = pickle.load(f)
    vocab: IVocab = ptb.load_data(use_cache=False)

    lstm_model_params = LSTMModelParams(
        vocab_size=10000, wordvec_size=650, hiddent_size=650, words=vocab.get_words()
    )
    lstm_model_params.embedding1_wx = params[0]
    lstm_model_params.lstm1_wx = params[1]
    lstm_model_params.lstm1_wh = params[2]
    lstm_model_params.lstm1_wb = params[3]
    lstm_model_params.lstm2_wx = params[4]
    lstm_model_params.lstm2_wh = params[5]
    lstm_model_params.lstm2_wb = params[6]
    lstm_model_params.affine1_wb = params[8]
with open(n_pkl_path2, "wb") as f:
    pickle.dump(lstm_model_params, f)
    pass

with open(n_pkl_path2, "rb") as f:
    lstm_model_params: LSTMModelParams = pickle.load(f)
    model = LSTMModel(lstm_model_params)
    a = 10
