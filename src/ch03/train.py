import sys

sys.path.append("..")
from src.common.interfaces.IVocab import IVocab
from src.dataset.ptb import ptb
from src.common.models import SimpleCbowModel
from src.common.optimizers import AdamOptimizer
from common.trainers import Trainer
from pympler import asizeof

hidden_size = 5
max_epoch = 60
batch_size = 3

vocab: IVocab = ptb.load_data(use_cache=False, data_size=10000)
vocab.set_current_matrix(window_size=1)
# TODO:vocab获取训练数据会很大，数据也不需要存储为one-hot变量
context_words, center_words = vocab.get_word2vec_data(use_onehot=True, window_size=1)
print("context words", asizeof.asizeof(context_words) / 1024**2, context_words.shape)
print("center words", asizeof.asizeof(center_words) / 1024**2, center_words.shape)
model = SimpleCbowModel(vocab.get_size(), hidden_size)
optimizer = AdamOptimizer()
trainer = Trainer(model, optimizer)
trainer.train(context_words, center_words, max_epoch, batch_size)
print(f"训练后的词向量矩阵：{model._W_in}")
trainer.plot(use_ppl=True)
