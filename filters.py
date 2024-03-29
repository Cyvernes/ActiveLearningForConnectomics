from __future__ import annotations
import numpy as np
from tools import *
from plot_tools import *
import scipy.ndimage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Learners import ActiveLearningSAM


def evidenceSmallerOrEqualToZero(learner : ActiveLearningSAM) -> np.ndarray:
    """This is an auxiliary function for filtering function. 
        It computes the region of interest for filtering.
        The region of interest is the region of the image in wich the evidence is smaller or equal to zero.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Region of interest
    :rtype: np.ndarray
    """

    blob = learner.evidence < 0
    blob = blob.astype("uint8")
    return blob


def NotInMasksFromSegmentationStrategy(learner : ActiveLearningSAM) -> np.ndarray:
    """This is an auxiliary function for filtering function. 
        It computes the region of interest for filtering.
        The region of interest is the region that is not segmented by the segmentation strategy

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Region of interest
    :rtype: np.ndarray
    """
    blob = learner.masks_from_single_seeds <= 0
    blob = blob.astype("uint8")
    return blob


def threshOnUncertainty(learner : ActiveLearningSAM) -> np.ndarray:
    """This is an auxiliary function for filtering function. 
        It computes the region of interest in filtering.
        The region of interest is defined by thresholds on the uncertainty map.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :return: Region of interest
    :rtype: np.ndarray
    """

    min_p_thresh = 0
    max_p_thresh = 0.7
    max_evidence_thresh = np.log(max_p_thresh / (1 - max_p_thresh))
    if min_p_thresh != 0:
        min_evidence_thresh = np.log(min_p_thresh / (1 - min_p_thresh))
    blob = learner.evidence < max_evidence_thresh
    if min_p_thresh != 0:
        blob[learner.evidence < min_evidence_thresh] = 0
    blob = blob.astype("uint8")
    return blob


def filterTrivial(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
        This filter does nothing.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    return arr


def hardFilter(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Every values that are not in the region of interest is set to - infinity.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    blob = learner.filtering_aux_function(learner)
    arr[np.logical_not(blob)] = -np.inf
    return arr


def filterWithDist(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values are changed accoring to the distance from the border of the region of interest.
       A sigmoid function is applied so that the filter is continuous.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    blob = learner.filtering_aux_function(learner)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - 10))
    arr[np.logical_not(blob)] = -np.inf
    return arr


def filterWithDistWithBorder(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values are changed accoring to the distance from the border of the region of interest or from the brder of the image.
       A sigmoid function is applied so that the filter is continuous.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    blob = learner.filtering_aux_function(learner)
    a, b = blob.shape
    cv2.rectangle(blob, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - 10))
    arr[np.logical_not(blob)] = -np.inf
    return arr


def filterWithPercentile(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values greater than the 80th percentile are set to 0

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    percentile_thresh = 80
    arr[arr >= np.percentile(arr, percentile_thresh)] = 0
    return arr


def filterWithDistSkeleton(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values are changed accoring to the skeleton of the distance from the border

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    evidence = learner.evidence
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh / (1 - p_thresh))
    skeleton_mask = skeleton(evidence < evidence_thresh)
    return np.multiply(skeleton_mask, arr)


def filterGaussianDistFromKnownSeeds(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values are changed accoring to the gaussian distance from the already annotated points.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    filter_arr = np.zeros_like(arr)
    x_coords, y_coords = zip(*learner.input_points)
    filter_arr[y_coords, x_coords] = 1
    filter_arr = scipy.ndimage.gaussian_filter(filter_arr, sigma=100)
    maxx = np.max(filter_arr)
    filter_arr = filter_arr / maxx if maxx != 0 else filter_arr
    arr = np.multiply(arr, 1 - filter_arr)
    return arr


def filterDistFromKnownSeeds(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       Values are changed accoring to the distance from already annotated points.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    blob = np.ones_like(arr)
    x_coords, y_coords = zip(*learner.input_points)
    blob[y_coords, x_coords] = 0
    dist = cv2.distanceTransform(blob.astype("uint8"), cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - 30))
    return arr


def HybridGDFKS_hard(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       This is an hybrid filter. It calls filterGaussianDistFromKnownSeeds and hardFilter.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    arr = filterGaussianDistFromKnownSeeds(learner, arr)
    arr = hardFilter(learner, arr)
    return arr


def HybridDFKS_hard(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       This is an hybrid filter. It calls filterDistFromKnownSeeds and hardFilter.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    arr = filterDistFromKnownSeeds(learner, arr)
    arr = hardFilter(learner, arr)
    return arr


def HybridGDFKS_Dist(learner : ActiveLearningSAM, arr: np.ndarray) -> np.ndarray:
    """This function is a filter. It changes values in the array arr.
       This is an hybrid filter. It calls filterGaussianDistFromKnownSeeds and filterWithDist.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param arr: Array to be filtered
    :type arr: np.ndarray
    :return: Filtered array
    :rtype: np.ndarray
    """
    arr = filterGaussianDistFromKnownSeeds(learner, arr)
    arr = filterWithDist(learner, arr)
    return arr
