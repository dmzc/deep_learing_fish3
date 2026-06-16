from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"


class DecoderType(Enum):
    AUTO = "auto"
    PEEKY = "peeky"
    ATTENTION = "attention"


class EncoderType(Enum):
    AUTO = "auto"
    ATTENTION = "attention"
