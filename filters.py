import numpy as np
from tools import *


def filterTrivial(arr: np.ndarray, evidence = None) -> np.ndarray:
    return(arr)

def filterWithDist(arr : np.ndarray, evidence : np.ndarray) -> np.ndarray:
    min_p_thresh = 0.001
    max_p_thresh = 0.7
    max_evidence_thresh = np.log(max_p_thresh/ (1 - max_p_thresh))
    min_evidence_thresh = np.log(min_p_thresh/ (1 - min_p_thresh))
    blob =  evidence < max_evidence_thresh
    blob[evidence > min_evidence_thresh] = 0
    blob = blob.astype("uint8")
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - np.max(dist)/3))
    return(arr)

def filterWithDistWithBorder(arr : np.ndarray, evidence : np.ndarray) -> np.ndarray:
    min_p_thresh = 0.001
    max_p_thresh = 0.7
    max_evidence_thresh = np.log(max_p_thresh/ (1 - max_p_thresh))
    min_evidence_thresh = np.log(min_p_thresh/ (1 - min_p_thresh))
    blob =  evidence < max_evidence_thresh
    blob[evidence > min_evidence_thresh] = 0
    blob = blob.astype("uint8")
    a, b = blob.shape
    cv2.rectangle(blob, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    arr = np.multiply(arr, sigmoid(dist - np.max(dist)/3))
    return(arr)

def filterWithPercentile(arr : np.ndarray, evidence = None) -> np.ndarray:
    percentile_thresh = 80
    arr[arr >= np.percentile(arr, percentile_thresh )] = 0
    return(arr)

def filterWithDistSkeleton(arr : np.ndarray, evidence : np.ndarray) -> np.ndarray:
    p_thresh = 0.95
    evidence_thresh = np.log(p_thresh/(1 - p_thresh))
    skeleton_mask = skeleton(evidence < evidence_thresh)
    return(np.multiply(skeleton_mask, arr))