import numpy as np
from tools import *
from filters import *


def ArgmaxEvInSESeeds(learner) -> list:
    learner.SE_Seeds = sorted(
        learner.SE_Seeds,
        key=lambda x: getValueinArrFromInputFormat(learner.evidence, x),
    )
    next_seed = learner.SE_Seeds.pop(-1)
    return [next_seed]


def ArgmaxUncertaintyInSESeeds(learner) -> list:
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    learner.SE_Seeds = sorted(
        learner.SE_Seeds, key=lambda x: getValueinArrFromInputFormat(uncertainty, x)
    )
    next_seed = learner.SE_Seeds.pop(-1)
    return [next_seed]


def ArgmaxDist(learner) -> list:
    p_thresh = 0.8
    evidence_thresh = np.log(p_thresh / (1 - p_thresh))
    uncertainMask = learner.evidence < evidence_thresh
    next_seed = swap(findVisualCenter(uncertainMask))
    return [next_seed]


def ArgmaxUncertainty(learner) -> list:
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    cx, cy = np.unravel_index(np.argmax(uncertainty), learner.evidence.shape)
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


def ArgmaxEvidence(learner) -> list:
    if learner.filtering_function != hardFilter:
        Warning.warn(
            "Applying a smooth filter may not work for argmaxEvidence because evidence can be non positive"
        )
    evidence = learner.filtering_function(learner, learner.evidence)
    cx, cy = np.unravel_index(np.argmax(evidence), learner.evidence.shape)
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


# similar to argmax evidence but the metrics is positive so the filters behave correctly
def ArgmaxForegroundProbability(learner) -> list:
    foreground_probability = sigmoid(learner.evidence)
    foreground_probability = learner.filtering_function(learner, foreground_probability)
    cx, cy = np.unravel_index(
        np.argmax(foreground_probability), foreground_probability.shape
    )
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


def ArgmaxUncertaintyPathDist(learner) -> list:
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh / (1 - p_thresh))

    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    thresh = learner.uncertainty_function(evidence_thresh)
    distances = UncertaintyPathDist(uncertainty, learner.evidence, thresh)
    cx, cy = np.unravel_index(np.argmax(distances), distances.shape)
    next_seed = [cy, cx]
    return [next_seed]
