import numpy as np
from tools import *


def singleStrat(learner) -> int:
    return 0


def changeAtFirstMito(learner) -> int:
    return int((learner.current_strategy_idx == 1) or (True in learner.input_labels))


def changeAfterAGivenAmountOfSeed(learner, nb_seeds) -> int:
    return int(
        (learner.current_strategy_idx == 1)
        or (
            len(learner.input_labels) >= (nb_seeds // 3)
            and (True in learner.input_labels)
        )
    )
