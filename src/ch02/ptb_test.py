import sys

sys.path.append("..")
from src.dataset.ptb import ptb
from src.common.interfaces.IVocab import IVocab
from src.common.util import get_similar_words


if __name__ == "__main__":
    vocab: IVocab = ptb.load_data()
    vocab.set_current_matrix(use_SVD=True, svd_vector_size=100)
    query_words = ["you", "year", "car", "toyota"]
    for query_word in query_words:
        get_similar_words(query_word, vocab, 5)
        pass
