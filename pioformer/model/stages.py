from enum import IntEnum
from typing import Union


class TrainingStage(IntEnum):
    CTN = 1
    TPN = 2
    PRN = 3

    @classmethod
    def parse(cls, value: Union[int, str, "TrainingStage"]) -> "TrainingStage":
        if isinstance(value, cls):
            return value
        return cls(int(value))
