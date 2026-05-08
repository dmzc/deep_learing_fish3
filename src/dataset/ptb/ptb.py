import pickle
import re
from pathlib import Path
from src.common.vocab import Vocab
from common.enums.dataset_type import DatasetType

ptb_dir = Path(__file__).parent
# vocab_path = ptb_dir / "ptb.vocab.pkl"


def load_data(type: DatasetType = DatasetType.TRAIN) -> Vocab:
    vocab_file = ptb_dir / f"{type.value}.vocab.pkl"
    if Path.exists(vocab_file):
        with open(vocab_file, "rb") as fh:
            vocab: Vocab = pickle.load(fh)
        return vocab
    data_file = ptb_dir / f"ptb.{type.value}.txt"

    with open(data_file, "r", encoding="utf-8") as fh:
        text = _clean(fh.read())
        vocab = Vocab()
        # TODO:这里不能在拼接而成的大句子里构建共现矩阵，而是应该在每个句子内滑动
        vocab.build(text)
        with open(vocab_file, "wb") as fh:
            pickle.dump(vocab, fh)
    return vocab


def _clean(text: str) -> str:

    # 1. 替换所有换行为 <eos>
    text = re.sub(r"[\r\n]+", " <eos> ", text)

    # 2. 替换所有空白符（\t \xa0 \u200b 全半角空格等）→ 普通空格
    text = re.sub(r"\s+", " ", text)

    # 3. 干掉所有 ASCII 控制字符（\x00~\x1f）
    text = re.sub(r"[\x00-\x1f]", "", text)

    # 4. 干掉所有 Unicode 不可见控制字符
    text = re.sub(r"[\u200b\u200c\u200d\u00a0\uFEFF]", "", text)

    # 5. 清理多余空格 + 首尾空格
    text = " ".join(text.strip().split())
    return text
