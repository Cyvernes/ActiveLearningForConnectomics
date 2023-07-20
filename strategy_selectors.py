import numpy as np
from tools import *
from Learners import ActiveLearningSAM


def singleStrat(learner: ActiveLearningSAM) -> int:
    """This function encodes the trivial strategy choice. It should be used when only one strategy is used. 

    Args:
        learner (ActiveLearningSAM): Learner

    Returns:
        int: The strategy to use 
    """
    return 0


def changeAtFirstMito(learner : ActiveLearningSAM) -> int:
    """This function encodes strategy selector. The strategy is changed when the first mitochondrion is found. 

    Args:
        learner (ActiveLearningSAM): Learner

    Returns:
        int: The strategy to use 
    """
    return int((learner.current_strategy_idx == 1) or (True in learner.input_labels))


def changeGivenAmountOfSeenMito(learner : ActiveLearningSAM) -> int:
    """This function encodes strategy selector. The strategy is changed when a specific number of mitochondria are found.
    The treshold is directly defined in the code.

    Args:
        learner (ActiveLearningSAM): Learner

    Returns:
        int: The strategy to use 
    """
    thresh = 7
    return int((learner.current_strategy_idx == 1) or (sum(learner.input_labels)) >= thresh)


def changeAfterAGivenAmountOfSeed(learner : ActiveLearningSAM) -> int:
    """This function encodes strategy selector. The strategy is changed when a specific number of seeds have been given.
    The treshold is directly defined in the code.

    Args:
        learner (ActiveLearningSAM): Learner

    Returns:
        int: The strategy to use 
    """
    thresh = 35
    return int(
        (learner.current_strategy_idx == 1)
        or (len(learner.input_labels) >= thresh and (True in learner.input_labels))
    )
