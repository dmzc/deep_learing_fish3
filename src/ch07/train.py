from src.dataset.sequence import sequence
from src.common.models import Seq2SeqModel
from src.common.optimizers import AdamOptimizer
from src.common.trainers import Trainer
from src.common.enums import DecoderType
from src.common.util import eval_seq2seq
import numpy as np
import matplotlib.pyplot as plt

data = sequence.load_data()
(x_train, t_train), (x_test, t_test) = data[0], data[1]
char_to_id, id_to_char = sequence.get_vocab()

use_reverse = True
if use_reverse:  # 只反转encoder的输入
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

use_peeky = True

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 30
max_grad = 5.0

decoder_type: DecoderType = None

if use_peeky:
    decoder_type = DecoderType.PEEKY
# acc_name = "_"
# if use_reverse:
#     acc_name += "use_reverse"
# if use_peeky:
#     acc_name += "use_peekly"


model = Seq2SeqModel(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    wordvec_size=wordvec_size,
    decoder_type=decoder_type,
)

optimizer = AdamOptimizer()

trainer = Trainer(model=model, optimizer=optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.train(
        x=x_train, t=t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad
    )
    correct_num = 0
    all_test_num = len(x_test)
    for index in range(all_test_num):
        question, answer = x_test[[index]], t_test[[index]]
        correct_num += eval_seq2seq(
            model,
            question=question,
            answer=answer,
            id_to_char=id_to_char,
            char_to_id=char_to_id,
            verbose=index < 10,
            is_reverse=use_reverse,
        )

    acc = float(correct_num) / all_test_num
    acc_list.append(acc)
    print("val acc %.3f%%" % (acc * 100))

# 绘制图形
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.show()
