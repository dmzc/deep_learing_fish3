from enum import Enum


class SimilarStrategy(Enum):
    CO = "co"
    PPMI = "ppmi"
    PPMI_SVD = "ppmi_svd"
