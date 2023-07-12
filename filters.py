import numpy as np
from tools import *
from plot_tools import *


def evidenceSmallerOrEqualToZero(learner) -> np.ndarray:
    blob = learner.evidence < 0
    blob = blob.astype("uint8")
    return(blob)

def NotInMasksFromOneSeedOneMask(learner) -> np.ndarray:
    blob = learner.masks_from_single_seeds != True
    blob = blob.astype("uint8")
    return(blob)

def threshOnUncertainty(learner) -> np.ndarray:
    min_p_thresh = 0
    max_p_thresh = 0.7
    max_evidence_thresh = np.log(max_p_thresh/ (1 - max_p_thresh))
    if min_p_thresh != 0:
        min_evidence_thresh = np.log(min_p_thresh/ (1 - min_p_thresh))
    blob = learner.evidence < max_evidence_thresh
    if min_p_thresh != 0:
        blob[learner.evidence > min_evidence_thresh] = 0
    blob = blob.astype("uint8")
    return(blob)
    

def filterTrivial(learner, arr: np.ndarray) -> np.ndarray:
    return(arr)

def filterWithDist(learner, arr : np.ndarray) -> np.ndarray:
    blob = learner.filtering_aux_function(learner)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - 10))
    return(arr)

def filterWithDistWithBorder(learner, arr : np.ndarray) -> np.ndarray:
    blob = learner.filtering_aux_function(learner)
    a, b = blob.shape
    cv2.rectangle(blob, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - np.max(dist)/3))
    return(arr)

def filterWithPercentile(learner, arr : np.ndarray) -> np.ndarray:
    percentile_thresh = 80
    arr[arr >= np.percentile(arr, percentile_thresh )] = 0
    return(arr)

def filterWithDistSkeleton(learner, arr : np.ndarray) -> np.ndarray:
    evidence = learner.evidence
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh/(1 - p_thresh))
    skeleton_mask = skeleton(evidence < evidence_thresh)
    return(np.multiply(skeleton_mask, arr))