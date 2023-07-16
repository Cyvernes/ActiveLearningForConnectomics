import numpy as np
from tools import *


def singleStrat(learner) -> int:
    return 0


def changeAtFirstMito(learner) -> int:
    return int((learner.current_strategy_idx == 1) or (True in learner.input_labels))


def changeGivenAmountOfSeenMito(learner) -> int:
    return int((learner.current_strategy_idx == 1) or (sum(learner.input_labels)) >= 2)


def changeAfterAGivenAmountOfSeed(learner) -> int:
    return int(
        (learner.current_strategy_idx == 1) or (len(learner.input_labels) >= 35 and (True in learner.input_labels))
    )
