import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import heapq
from scipy.signal import convolve2d

def centroid(mask : np.array):
    a, b = np.where(mask)
    return(int(np.mean(a)), int(np.mean(b)))

def swap(t : list):
    return([t[1], t[0]])

def IoU(mask1, mask2):
    intersection = np.sum(np.bitwise_and(mask1, mask2))
    union = np.sum(np.bitwise_or(mask1, mask2))
    return(intersection / union)

def FP(mask1, mask2):
    h, w = mask1.shape
    sum = np.sum(np.logical_and(np.bitwise_xor(mask1, mask2), mask1))
    return(sum /(h*w))

def FN(mask1, mask2):
    h, w = mask1.shape
    sum = np.sum(np.logical_and(np.bitwise_xor(mask1, mask2), mask2))
    return(sum /(h*w))

def getLabel(seed, GT_mask): #seed should be given in the input format and not in the image format
    return(GT_mask[seed[1], seed[0]])

def getEvidence(evidence, seed): #seed should be given in the input format and not in the image format
    return(evidence[seed[1], seed[0]])

def sigmoid(x):
    return(1 /(1 + np.exp(-x)))

def findVisualCenter(mask):
    new_mask = mask.copy().astype("uint8")
    a, b = new_mask.shape
    cv2.rectangle(new_mask, (0, 0), (a - 1, b - 1), (0), 1)
    dist = cv2.distanceTransform(new_mask, cv2.DIST_L2, 3)
    return(np.unravel_index(np.argmax(dist), dist.shape))

def uncertaintyKL(Ev):
    abs_Ev = np.abs(Ev)
    return(np.log( 1 + np.exp(-abs_Ev)))

def uncertaintyH(Ev):
    p = sigmoid(Ev)
    one_minus_p = 1-p
    rep = np.zeros_like(p)
    where_to_compute = np.logical_and(p > 0, p < 1)
    rep[where_to_compute] = - np.multiply(p[where_to_compute], np.log(p[where_to_compute])) - np.multiply(one_minus_p[where_to_compute], np.log(one_minus_p[where_to_compute]))
    return(rep)

def findNewSeedWithUncertaintyPathDist(evidence : np.array, thresh : float):
    uncertainty = uncertaintyH(evidence)
    uncertain_mask = uncertainty > uncertaintyKL(thresh)
    uncertain_mask = evidence < thresh
    certain_mask = np.logical_not(uncertain_mask)
    distances = np.zeros_like(evidence)
    seen_points = np.zeros_like(evidence)
    seen_points[certain_mask] = 1
    h, w = evidence.shape
    #find all 1 in uncertain mask adjacent to zeros
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    count = convolve2d(uncertain_mask, kernel, mode = "same", fillvalue = 0)
    points_x, points_y = np.where(np.bitwise_and(uncertain_mask, count < 8))
    points_to_see = [(uncertainty[point_x, point_y], (point_x, point_y)) for point_x, point_y in zip(points_x, points_y)]
    heapq.heapify(points_to_see)
    while len(points_to_see) != 0:#proof of correctness: weight is increasing
        weight, point = heapq.heappop(points_to_see)
        if seen_points[point[0], point[1]] == 0:
            distances[point[0], point[1]] = weight
            seen_points[point[0], point[1]] = 1
        else:
            continue
        for dx, dy in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            if (0 <= point[0] + dx < h) and (0 <= point[1] + dy < w) and (uncertain_mask[point[0] + dx, point[1] + dy] and seen_points[point[0] + dx, point[1] + dy] == 0):
                potential_weight = np.sqrt(np.abs(dx) + np.abs(dy)) * uncertainty[point[0] + dx, point[1] + dy] + distances[point[0], point[1]]
                if distances[point[0] + dx, point[1] + dy] == 0 or potential_weight < distances[point[0] + dx, point[1] + dy]:#tiny optimization
                    distances[point[0] + dx, point[1] + dy] = potential_weight
                    heapq.heappush(points_to_see, (potential_weight, (point[0] + dx, point[1] + dy)))
    
    
    return(np.unravel_index(np.argmax(distances), distances.shape))
    
    
    