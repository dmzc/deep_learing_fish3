import sys

sys.path.append("..")
from src.common.interfaces.IVocab import IVocab
from src.dataset.ptb import ptb
from pympler import asizeof
from src.common.models import CbowModel
from src.common.optimizers import AdamOptimizer
from common.trainers import Trainer
import pickle
import numpy as np

hidden_size = 100
max_epoch = 10
batch_size = 128
window_size=5

vocab: IVocab = ptb.load_data(use_cache=False)
vocab.set_current_matrix(window_size=window_size)
context_words, center_words = vocab.get_word2vec_data(use_onehot=False, window_size=window_size)
# ptb 90w字的语料，使用one-host形式，center words 8GB,Context Words 17GB。使用id形式，只有20MB
print(
    f"context words:\n Size: {asizeof.asizeof(context_words) / 1024**2} MB\nShape:{context_words.shape}"
)
print(
    f"center words:\n Size: {asizeof.asizeof(center_words) / 1024**2} MB\nShape:{center_words.shape}"
)
model = CbowModel(vocab=vocab, hidden_size=hidden_size)
optimizer = AdamOptimizer()
trainer = Trainer(model, optimizer)
trainer.train(context_words, center_words, max_epoch, batch_size)
print(f"训练后的词向量矩阵：{model._W_in}")
params = {}
params["word_vecs"] = model._W_in.astype(np.float16)
pkl_file = "cbow_params.pkl"  # or 'skipgram_params.pkl'
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
trainer.plot(use_ppl=True)
