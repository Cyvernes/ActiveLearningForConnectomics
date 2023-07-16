import numpy as np
from tools import *
from plot_tools import *
import scipy.ndimage


def evidenceSmallerOrEqualToZero(learner) -> np.ndarray:
    blob = learner.evidence < 0
    blob = blob.astype("uint8")
    return blob


def NotInMasksFromOneSeedOneMask(learner) -> np.ndarray:
    blob = learner.masks_from_single_seeds <= 0
    blob = blob.astype("uint8")
    return blob


def threshOnUncertainty(learner) -> np.ndarray:
    min_p_thresh = 0
    max_p_thresh = 0.7
    max_evidence_thresh = np.log(max_p_thresh / (1 - max_p_thresh))
    if min_p_thresh != 0:
        min_evidence_thresh = np.log(min_p_thresh / (1 - min_p_thresh))
    blob = learner.evidence < max_evidence_thresh
    if min_p_thresh != 0:
        blob[learner.evidence > min_evidence_thresh] = 0
    blob = blob.astype("uint8")
    return blob


def filterTrivial(learner, arr: np.ndarray) -> np.ndarray:
    return arr


def hardFilter(learner, arr: np.ndarray) -> np.ndarray:
    blob = learner.filtering_aux_function(learner)
    arr[np.logical_not(blob)] = -np.inf
    return arr


def filterWithDist(learner, arr: np.ndarray) -> np.ndarray:
    blob = learner.filtering_aux_function(learner)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - 10))
    return arr


def filterWithDistWithBorder(learner, arr: np.ndarray) -> np.ndarray:
    blob = learner.filtering_aux_function(learner)
    a, b = blob.shape
    cv2.rectangle(blob, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - np.max(dist) / 3))
    return arr


def filterWithPercentile(learner, arr: np.ndarray) -> np.ndarray:
    percentile_thresh = 80
    arr[arr >= np.percentile(arr, percentile_thresh)] = 0
    return arr


def filterWithDistSkeleton(learner, arr: np.ndarray) -> np.ndarray:
    evidence = learner.evidence
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh / (1 - p_thresh))
    skeleton_mask = skeleton(evidence < evidence_thresh)
    return np.multiply(skeleton_mask, arr)


def filterGaussianDistFromKnownSeeds(learner, arr: np.ndarray) -> np.ndarray:
    filter_arr = np.zeros_like(arr)
    x_coords, y_coords = zip(*learner.input_points)
    filter_arr[y_coords, x_coords] = 1
    filter_arr = scipy.ndimage.gaussian_filter(filter_arr, sigma=50)
    maxx = np.max(filter_arr)
    filter_arr = filter_arr / maxx if maxx != 0 else filter_arr
    arr = np.multiply(arr, 1 - filter_arr)
    return arr


def HybridGDFKS_hard(learner, arr: np.ndarray) -> np.ndarray:
    arr = filterGaussianDistFromKnownSeeds(learner, arr)
    arr = hardFilter(learner, arr)
    return arr


def HybridGDFKS_Dist(learner, arr: np.ndarray) -> np.ndarray:
    arr = filterGaussianDistFromKnownSeeds(learner, arr)
    arr = filterWithDist(learner, arr)
    return arr
