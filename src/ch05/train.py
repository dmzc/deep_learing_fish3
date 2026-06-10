from src.common.trainers import RNNTrainer
from src.common.optimizers import SGDOptimizer
from src.dataset.ptb import ptb
from src.common.models import SimpleRNNModel
from src.common.vocab import Vocab

# 设定超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN的隐藏状态向量的元素个数
time_size = 5  # RNN的展开大小
lr = 0.1
max_epoch = 100

# 读入训练数据
vocab: Vocab = ptb.load_data(use_cache=False, data_size=1000)
xs, ts = vocab.get_rnn_data()
model = SimpleRNNModel(vocab.get_size(), wordvec_size, hidden_size)
optimizer = SGDOptimizer(lr)
trainer = RNNTrainer(model, optimizer)
trainer.train(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()
