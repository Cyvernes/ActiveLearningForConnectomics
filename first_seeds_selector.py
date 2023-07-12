from typing import List, Tuple

def popLastSESeeds(learner) -> List[Tuple[int, int]]:
    SE_mask = learner.SE_masks.pop(-1)['segmentation'] 
    first_seed = learner.SE_Seeds.pop(-1)
    return([first_seed], SE_mask)

def allSESeeds(learner) -> List[Tuple[int, int]]:
    SE_mask = learner.SE_masks.pop(-1)['segmentation'] 
    first_seeds = learner.SE_Seeds
    learner.SE_seeds = []
    return(first_seeds, SE_mask)

def allForegroundSESeeds(learner) -> List[Tuple[int, int]]:
    SE_mask = learner.SE_masks.pop(-1)['segmentation'] 
    first_seeds = learner.SE_Seeds
    learner.SE_seeds = []
    return(first_seeds, SE_mask)
    