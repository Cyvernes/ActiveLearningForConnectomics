import numpy as np
from tools import *

def ArgmaxEvInSESeeds(learner) -> list:
    learner.SE_Seeds = sorted(learner.SE_Seeds, key = lambda x : getValueinArrFromInputFormat(learner.evidence, x))
    new_seed = learner.SE_Seeds.pop(-1)
    return([new_seed])


def ArgmaxDist(learner)  -> list:
    p_thresh = 0.8
    evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
    uncertainMask = learner.evidence < evidence_thresh
    new_seed = swap(findVisualCenter(uncertainMask))
    return([new_seed])

def ArgmaxUncertainty(learner)  -> list:
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(uncertainty, learner.evidence)
    cx, cy = np.unravel_index(np.argmax(uncertainty), learner.evidence.shape)
    new_seed = [cy, cx]#swap cx and cy to meet input format
    return([new_seed])

def ArgmaxUncertaintyPathDist(learner)  -> list:
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
    
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(uncertainty, learner.evidence)
    thresh = learner.uncertainty_function(evidence_thresh)
    distances = UncertaintyPathDist(uncertainty, learner.evidence, thresh)
    cx, cy = np.unravel_index(np.argmax(distances), distances.shape)
    new_seed = [cy, cx]
    return([new_seed])