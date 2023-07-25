from typing import List, Tuple
from tools import *
from Learners import ActiveLearningSAM


def popLastSESeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        The last seed from SE seeds is selected.
        learner.SE_seeds is updated accordingly.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seed = learner.SE_Seeds.pop(-1)
    return ([first_seed], SE_mask)


def allSESeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        All SE seeds are selected.
        learner.SE_seeds is updated accordingly.


    """
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = learner.SE_Seeds
    learner.SE_Seeds = []
    return (first_seeds, SE_mask)


def allForegroundSESeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        All foreground SE seeds are selected.
        learner.SE_seeds is updated accordingly.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    assert (
        learner.need_ground_truth
    ), "Learner does not have access to Ground Truth, maybe you should use the PseudoActiveLearningSAM class instead"
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = [
        s for s in learner.SE_Seeds if getValueinArrFromInputFormat(learner.GT_mask, s)
    ]
    learner.SE_Seeds = [
        s
        for s in learner.SE_Seeds
        if not getValueinArrFromInputFormat(learner.GT_mask, s)
    ]
    return (first_seeds, SE_mask)


def aGivenAmountOfForegroundSESeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        A specific number of foreground SE seeds is selected.
        learner.SE_seeds is updated accordingly.
    
    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    amount = 4
    assert (
        learner.need_ground_truth
    ), "Learner does not have access to Ground Truth, maybe you should use the PseudoActiveLearningSAM class instead"
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = [
        s for s in learner.SE_Seeds if getValueinArrFromInputFormat(learner.GT_mask, s)
    ]
    first_seeds = first_seeds[-min(len(first_seeds), amount):]
    learner.SE_Seeds = [s for s in learner.SE_Seeds if not s in first_seeds]
    return (first_seeds, SE_mask)

def aGivenAmountOfSESeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        A specific number of SE seeds is selected.
        learner.SE_seeds is updated accordingly.
    
    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    amount = 4
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = learner.SE_Seeds[-min(len(first_seeds), amount):]
    learner.SE_Seeds = learner.SE_Seeds[0:- min(len(first_seeds), amount)]
    return (first_seeds, SE_mask)