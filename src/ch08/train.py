# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from src.dataset.sequence import sequence
from src.common.optimizers import AdamOptimizer
from src.common.trainers import Trainer
from src.common.util import eval_seq2seq
from src.common.models import Seq2SeqModel
from src.common.enums import DecoderType, EncoderType
from src.common.logger import getLogger
from time import time

logger = getLogger()

# 读入数据
(x_train, t_train), (x_test, t_test) = sequence.load_data()
use_reverse = True
if use_reverse:  # 只反转encoder的输入
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# (x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
char_to_id, id_to_char = sequence.get_vocab()

# 反转输入语句
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 设定超参数
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 25
max_grad = 5.0

# logger.info(f"vocab_size{vocab_size}\n")

model = Seq2SeqModel(
    vocab_size,
    wordvec_size,
    hidden_size,
    decoder_type=DecoderType.ATTENTION,
    encoder_type=EncoderType.ATTENTION,
)

optimizer = AdamOptimizer()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    start_time = time()
    last_loss = trainer.train(
        x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad
    )
    print("-" * 25 + f"第{epoch}轮开始" + "-" * 25)
    print(f"训练耗时：{time() - start_time}")
    print(f"训练损失：{last_loss}")

    if epoch % 5 == 0 or epoch == max_epoch - 1:
        start_time = time()
        correct_num = 0
        test_data_len = len(x_test)
        for index in range(test_data_len):
            question, answer = x_test[[index]], t_test[[index]]
            verbose = index < 10
            correct_num += eval_seq2seq(
                model, question, answer, id_to_char, verbose, is_reverse=use_reverse
            )

        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print(f"测试耗时：{time() - start_time}")
        print("准确率：%.3f%%" % (acc * 100))
        print("-" * 25 + f"第{epoch}轮结束" + "-" * 25)
        print("\n")


# 绘制图形
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(-0.05, 1.05)
plt.show()
