# coding: utf-8
import sys

sys.path.append("..")
from src.common.interfaces.IVocab import IVocab
from src.common.optimizers import SGDOptimizer
from src.common.trainers import RNNTrainer

# from common.util import eval_perplexity, to_gpu
from src.dataset.ptb import ptb
from src.common.models import LSTMModel, LSTMModelParams
from src.common.enums import DatasetType

# 设定超参数
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 10
max_grad = 0.25
dropout = 0.5

vocab: IVocab = ptb.load_data(use_cache=False)
valid_vocab = ptb.load_data(use_cache=False, type=DatasetType.VALID)
test_vocab = ptb.load_data(use_cache=False, type=DatasetType.TEST)


vocab_size = vocab.get_size()
xs, ts = vocab.get_rnn_data()

model = LSTMModel(
    LSTMModelParams(
        vocab_size=vocab_size,
        wordvec_size=wordvec_size,
        hiddent_size=hidden_size,
        dropout_ratio=dropout,
        words=vocab.get_words(),
    )
)
optimizer = SGDOptimizer(lr)
trainer = RNNTrainer(model, optimizer)

best_ppl = float("inf")
# TODO：cpu跑要两天，先不跑
for epoch in range(max_epoch):
    trainer.train(
        xs,
        ts,
        max_epoch=1,
        batch_size=batch_size,
        time_size=time_size,
        max_grad=max_grad,
    )

    model.reset_state()
    # ppl = eval_perplexity(model, corpus_val)
    # print("valid perplexity: ", ppl)

    # if best_ppl > ppl:
    #     best_ppl = ppl
    #     model.save_params()
    # else:
    #     lr /= 4.0
    #     optimizer.lr = lr

    # model.reset_state()
    # print("-" * 50)


# 基于验证数据进行评价
# model.reset_state()
# ppl_test = eval_perplexity(model, corpus_test)
# print("test perplexity: ", ppl_test)
