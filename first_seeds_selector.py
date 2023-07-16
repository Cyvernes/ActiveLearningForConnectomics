from typing import List, Tuple
from tools import *


def popLastSESeeds(learner) -> List[Tuple[int, int]]:
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seed = learner.SE_Seeds.pop(-1)
    return ([first_seed], SE_mask)


def allSESeeds(learner) -> List[Tuple[int, int]]:
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = learner.SE_Seeds
    learner.SE_seeds = []
    return (first_seeds, SE_mask)


def allForegroundSESeeds(learner) -> List[Tuple[int, int]]:
    assert (
        learner.need_ground_truth
    ), "Learner does not have access to Ground Truth, maybe you should use the PseudoActiveLearningSAM class instead"
    SE_mask = learner.SE_masks.pop(-1)["segmentation"]
    first_seeds = [s for s in learner.SE_Seeds if getValueinArrFromInputFormat(learner.GT_mask, s)]
    learner.SE_seeds = []
    return (first_seeds, SE_mask)
