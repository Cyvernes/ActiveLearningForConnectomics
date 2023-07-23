from __future__ import annotations
import numpy as np
from tools import *
from filters import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Learners import ActiveLearningSAM
    
def ArgmaxEvInSESeeds(learner : ActiveLearningSAM) -> list:
    """This function encodes a sampling strategy.
        The seed of maximum evidence in SE seeds is selected.
        learner.SE_seeds is updated accordingly

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Next points to annotated
    :rtype: list
    """
    evidence = learner.filtering_function(learner, learner.evidence)
    learner.SE_Seeds = sorted(
        learner.SE_Seeds,
        key=lambda x: getValueinArrFromInputFormat(evidence, x),
    )
    next_seed = learner.SE_Seeds.pop(-1)
    return [next_seed]


def ArgmaxUncertaintyInSESeeds(learner : ActiveLearningSAM) -> list:
    """This function encodes a sampling strategy.
        The seed of maximum uncertainty in SE seeds is selected.
        learner.SE_seeds is updated accordingly

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Next points to annotated
    :rtype: list
    """
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    learner.SE_Seeds = sorted(
        learner.SE_Seeds, key=lambda x: getValueinArrFromInputFormat(uncertainty, x)
    )
    next_seed = learner.SE_Seeds.pop(-1)
    return [next_seed]



def ArgmaxUncertainty(learner : ActiveLearningSAM) -> list:
    """This function encodes a sampling strategy.
        The point of maximum uncertainty is selected.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Next points to annotate
    :rtype: list
    """
    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    cx, cy = np.unravel_index(np.argmax(uncertainty), learner.evidence.shape)
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


def ArgmaxEvidence(learner : ActiveLearningSAM) -> list:
    """This function encodes a sampling strategy.
        The point of maximum evidence is selected.
        This function bears a strong resemblance to ArgmaxForegroundProbability.  
        If you intend to employ filters, it is advisable to opt for ArgmaxForegroundProbability.


    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Next points to annotate
    :rtype: list
    """
    if learner.filtering_function != hardFilter:
        Warning.warn(
            "Applying a smooth filter may not work for argmaxEvidence because evidence can be non positive"
        )
    evidence = learner.filtering_function(learner, learner.evidence)
    cx, cy = np.unravel_index(np.argmax(evidence), learner.evidence.shape)
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


def ArgmaxForegroundProbability(
    learner : ActiveLearningSAM,
) -> (
    list
):   
    """This function encodes a sampling strategy.
        The point of maximum foreground probability is selected.
        This function bears a strong resemblance to ArgmaxEvidence, with the notable distinction that filters are expected to function accurately in this case. 

    :param learner: Learner
    :type learner: ActiveLearningSAM,
    :return: Next points to annotate
    :rtype: list
    """
    foreground_probability = sigmoid(learner.evidence)
    foreground_probability = learner.filtering_function(learner, foreground_probability)
    cx, cy = np.unravel_index(
        np.argmax(foreground_probability), foreground_probability.shape
    )
    next_seed = [cy, cx]  # swap cx and cy to meet input format
    return [next_seed]


def ArgmaxUncertaintyPathDist(learner : ActiveLearningSAM) -> list:
    """This function encodes a sampling strategy.
        The point of maximum distance is selected.
        In this function the distance is the Uncertainty path distance.
        This function might be very slow. Some speed improvements should be made.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Next points to annotate
    :rtype: list
    """
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh / (1 - p_thresh))

    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    thresh = learner.uncertainty_function(evidence_thresh)
    distances = UncertaintyPathDist(uncertainty, learner.evidence, thresh)
    cx, cy = np.unravel_index(np.argmax(distances), distances.shape)
    next_seed = [cy, cx]
    return [next_seed]
