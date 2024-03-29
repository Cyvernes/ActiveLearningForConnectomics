import numpy as np
import cv2
import itertools
import heapq
import os
from scipy.signal import convolve2d
from tools import *
from typing import Tuple


def removeTooCloseSeeds(seeds: list) -> list:
    """This function remove points that are too close from each other, i.e. points whose distance is less than 10

    :param seeds: ist of points to process
    :type seeds: list
    :return: Processed list
    :rtype: list
    """

    points_that_need_to_be_deleted = []
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            if (seed_a[0] - seed_b[0]) ** 2 + (seed_a[1] - seed_b[1]) ** 2 <= 100:
                points_that_need_to_be_deleted.append(seed_a)
                break
    return [s for s in seeds if not s in points_that_need_to_be_deleted]


def centroid(mask: np.ndarray) ->  Tuple[int, int]:
    """Computes the centroid of a mask

    :param mask: Mask
    :type mask: np.ndarray
    :return: Centroid of the mask
    :rtype: Tuple[int, int]
    """

    a, b = np.where(mask)
    return (int(np.mean(a)), int(np.mean(b)))


def swap(t: list) -> list:
    """Swap coordinates of a point

    :param t: Point
    :type t: list
    :return: Swaped coordinates
    :rtype: list
    """
    return [t[1], t[0]]


def IoU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Intersection over Union of two masks.

    :param mask1: Mask 1
    :type mask1: np.ndarray
    :param mask2: Mask 2
    :type mask2: np.ndarray
    :return: Intersection over Union
    :rtype: float
    """
    intersection = np.sum(np.bitwise_and(mask1, mask2))
    union = np.sum(np.bitwise_or(mask1, mask2))
    return intersection / union

def DiceLoss(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Dice loss of two masks.

    :param mask1: Mask 1
    :type mask1: np.ndarray
    :param mask2: Mask 2
    :type mask2: np.ndarray
    :return: Dice loss 
    :rtype: float
    """
    intersection = np.sum(np.bitwise_and(mask1, mask2))
    denominator = np.sum(mask1) + np.sum(mask2)
    return 2*intersection / denominator

def FP(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the rate of false poitive according to mask2

    :param mask1: Mask
    :type mask1: np.ndarray
    :param mask2: ground Truth
    :type mask2: np.ndarray
    :return: False positive rate
    :rtype: float
    """
    h, w = mask1.shape
    sum = np.sum(np.logical_and(np.bitwise_xor(mask1, mask2), mask1))
    return sum / (h * w)


def FN(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the rate of false negative according to mask2

    :param mask1: Mask
    :type mask1: np.ndarray
    :param mask2: Ground truth
    :type mask2: np.ndarray
    :return: False negative rate
    :rtype: float
    """
    h, w = mask1.shape
    sum = np.sum(np.logical_and(np.bitwise_xor(mask1, mask2), mask2))
    return sum / (h * w)


def getLabel(
    seed: Tuple[int, int], GT_mask: np.ndarray
) -> bool: 
    """Retrieves the label of a point
        Seeds are given in the prompt format. 
        That's why coordinates are swaped to get the label in the GT format.

    :param seed: Seed
    :type seed: Tuple[int, int]
    :param GT_mask: Ground truth mask
    :type GT_mask: np.ndarray
    :return: Label of the seed
    :rtype: bool
    """
    return GT_mask[seed[1], seed[0]]


def getValueinArrFromInputFormat(
    arr: np.ndarray, seed: Tuple[int, int]
): 
    """Retrieves the value of a point in the array. 
        The point is supposed to be given in the prompt format.

    :param arr: Array in which to retrieve the value
    :type arr: np.ndarray
    :param seed: Coordinates
    :type seed: Tuple[int, int]
    :return: Value of the point in the array.
    :rtype: _type_
    """
    return arr[seed[1], seed[0]]


def sigmoid(x : np.ndarray) -> np.ndarray:
    """Computes the sigmoid of an array.

    :param x: array
    :type x: np.ndarray
    :return: sigmoid(array)
    :rtype: np.ndarray
    """
    return 1 / (1 + np.exp(-x))


def findVisualCenter(mask: np.ndarray) -> Tuple[int, int]:
    """Computes the visual center of a mask.

    :param mask: Mask
    :type mask: np.ndarray
    :return: Visual center of the mask
    :rtype: Tuple[int, int]
    """
    new_mask = mask.copy().astype("uint8")
    a, b = new_mask.shape
    cv2.rectangle(new_mask, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(new_mask, cv2.DIST_L2, 3)
    return np.unravel_index(np.argmax(dist), dist.shape)


def uncertaintyKL(Ev : np.ndarray) -> np.ndarray:
    """Computes the uncertainty of an array.
        The uncertainty function comes from the Kullback-Leibler divergence.

    :param Ev: Evidence map
    :type Ev: np.ndarray
    :return: Uncertainty map
    :rtype: np.ndarray
    """
    abs_Ev = np.abs(Ev)
    return np.log(1 + np.exp(-abs_Ev))


def uncertaintyH(Ev):
    """Computes the uncertainty of an array.
        The uncertainty function comes from the binary entropy.

    :param Ev: Evidence map
    :type Ev: _type_
    :return: Uncertainty map
    :rtype: _type_
    """
    p = sigmoid(Ev)
    one_minus_p = 1 - p
    rep = np.zeros_like(p)
    where_to_compute = np.logical_and(p > 0, p < 1)
    rep[where_to_compute] = -np.multiply(
        p[where_to_compute], np.log(p[where_to_compute])
    ) - np.multiply(
        one_minus_p[where_to_compute], np.log(one_minus_p[where_to_compute])
    )
    return rep


def UncertaintyPathDist(
    uncertainty: np.ndarray, evidence: np.ndarray, thresh: float
) -> np.ndarray:
    """Computes the uncertainty path distance of an uncertainty map
    Might be very slow and does not use filtering auxiliary function to compute the region of interest.

    :param uncertainty: Uncertainty map
    :type uncertainty: np.ndarray
    :param evidence: Evidence map
    :type evidence: np.ndarray
    :param thresh:  Threshold for the region of interest
    :type thresh: float
    :return: Uncertainty path distance map
    :rtype: np.ndarray
    """

    uncertain_mask = uncertainty > thresh
    uncertain_mask = evidence < thresh
    certain_mask = np.logical_not(uncertain_mask)
    distances = np.zeros_like(evidence)
    seen_points = np.zeros_like(evidence)
    seen_points[certain_mask] = 1
    h, w = evidence.shape
    # find all 1 in uncertain mask adjacent to zeros
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    count = convolve2d(uncertain_mask, kernel, mode="same", fillvalue=0)
    points_x, points_y = np.where(np.bitwise_and(uncertain_mask, count < 8))
    points_to_see = [
        (uncertainty[point_x, point_y], (point_x, point_y))
        for point_x, point_y in zip(points_x, points_y)
    ]
    heapq.heapify(points_to_see)
    while len(points_to_see) != 0:  # proof of correctness: weight is increasing
        weight, point = heapq.heappop(points_to_see)
        if seen_points[point[0], point[1]] == 0:
            distances[point[0], point[1]] = weight
            seen_points[point[0], point[1]] = 1
            for dx, dy in [
                (1, 1),
                (1, 0),
                (1, -1),
                (0, 1),
                (0, -1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
            ]:
                if (
                    (0 <= point[0] + dx < h)
                    and (0 <= point[1] + dy < w)
                    and (
                        uncertain_mask[point[0] + dx, point[1] + dy]
                        and seen_points[point[0] + dx, point[1] + dy] == 0
                    )
                ):
                    potential_weight = (
                        np.sqrt(np.abs(dx) + np.abs(dy))
                        * uncertainty[point[0] + dx, point[1] + dy]
                        + distances[point[0], point[1]]
                    )
                    if (
                        distances[point[0] + dx, point[1] + dy] == 0
                        or potential_weight < distances[point[0] + dx, point[1] + dy]
                    ):  # tiny optimization
                        distances[point[0] + dx, point[1] + dy] = potential_weight
                        heapq.heappush(
                            points_to_see,
                            (potential_weight, (point[0] + dx, point[1] + dy)),
                        )

    return distances


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Computes the skeleton of a mask
        This is done thaks to dist transform and laplacian analysis

    :param mask: Mask
    :type mask: np.ndarray
    :return: Skeleton of the mask
    :rtype: np.ndarray
    """
    mask = mask.astype("uint8")
    a, b = mask.shape
    cv2.rectangle(mask, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize=3)
    lap = -cv2.Laplacian(dist, ddepth=-1, ksize=3)
    lap[lap < 0] = 0
    lap = lap / np.max(lap)
    lap = np.multiply(lap, sigmoid(dist - np.max(dist) / 3))

    lap[lap < 0.2] = 0
    lap[lap >= 0.2] = 1
    return lap
