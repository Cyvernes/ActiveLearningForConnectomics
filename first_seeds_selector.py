from typing import List, Tuple
from tools import *
from Learners import ActiveLearningSAM
import cv2

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
        
        If there is not enough foreground seeds from SE_Seeds, then all foreground SE_Seeds are selected.
        
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
        
        If there is not enough points in SE_Seeds, then all SE_Seeds are selected.
        
        learner.SE_seeds is updated accordingly.
    
    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    amount = 4
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = learner.SE_Seeds[-min(len(learner.SE_Seeds), amount):]
    learner.SE_Seeds = learner.SE_Seeds[0:- min(len(learner.SE_Seeds), amount)]
    return (first_seeds, SE_mask)

def aGivenAmountOfForegroundSeeds(learner : ActiveLearningSAM) -> List[Tuple[int, int]]:
    """This function encodes a strategy to find the first point to annotate.
        A specific number of foreground seeds is selected.
        learner.SE_seeds is updated accordingly.
    
    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: List of first points to annotate.
    :rtype: List[Tuple[int, int]]
    """
    amount = 4
    contours, _ = cv2.findContours(learner.GT_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = [cv2.moments(cnt) for cnt in contours]
    moments.sort(key = lambda x : float(x['m00']))
    seeds = [[int(M['m10']/M['m00']), int(M['m01']/M['m00'])] for M in moments]
    first_seeds = seeds[-min(len(learner.SE_Seeds), amount):]
    
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    return (first_seeds, SE_mask)