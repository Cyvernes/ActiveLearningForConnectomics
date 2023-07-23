from __future__ import annotations
import numpy as np
from tools import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Learners import ActiveLearningSAM

def singleStrat(learner: ActiveLearningSAM) -> int:
    """This function encodes the trivial strategy choice. 
    It should be used when only one strategy is used. 

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Index of the strategy to use (0 for this function)
    :rtype: int
    """

    return 0


def changeAtFirstMito(learner : ActiveLearningSAM) -> int:
    """This function encodes a strategy selector. 
    The strategy is changed when the first mitochondrion is found.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Index of the strategy to use
    :rtype: int
    """
    return int((learner.current_strategy_idx == 1) or (True in learner.input_labels))


def changeAfterASpecificNumberOfSeenMito(learner : ActiveLearningSAM) -> int:
    """This function encodes strategy selector. 
    The strategy is changed when a specific number of mitochondria are found.
    The treshold is directly defined in the code

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Index of the strategy to use
    :rtype: int
    """
    thresh = 7
    return int((learner.current_strategy_idx == 1) or (sum(learner.input_labels)) >= thresh)


def changeAfterASpecificNumberOfSeed(learner : ActiveLearningSAM) -> int:
    """This function encodes strategy selector. 
    The strategy is changed when a specific number of seeds have been given.
    The treshold is directly defined in the code.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Index of the strategy to use
    :rtype: int
    """
    thresh = 35
    return int(
        (learner.current_strategy_idx == 1)
        or (len(learner.input_labels) >= thresh and (True in learner.input_labels))
    )
