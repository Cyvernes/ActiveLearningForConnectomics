import torch
import torchvision
import numpy as np
import cv2
import sys
import json
import time
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import random
import matplotlib.pyplot as plt
from tools import *

class ActiveLearningSAM:
    def __init__(self, model, image = None, mask_generator = None ):
        self.model = model
        self.image = image
        self.cp_mask = None #Current predicted mask
        self.needGroundTruth = False
        
        if mask_generator:
            self.mask_generator = mask_generator
        else: #Instancing a mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model = self.model,
                points_per_side = 64,
                points_per_batch = 64,
                pred_iou_thresh = 0.95,
                stability_score_thresh = 0.92,
                crop_n_layers = 0,
                crop_n_points_downscale_factor = 2,
                min_mask_region_area = 0,  # Requires open-cv to run post-processing
            )
            
        #Instancing a predictor 
        self.predictor = predictor = SamPredictor(self.model)
        self.evidence = None
        self.look_for_first_GT_mitochondria = True
    
    def setData(self, image):
        self.image = image
        self.predictor.set_image(image)
        self.cp_mask = None
    
    def findFirstSeed(self):#masks from Segment Everything are in the same format as the image
        SE_masks = sorted(self.mask_generator.generate(self.image), key = lambda mask: mask['predicted_iou'])#masks from segement every thing
        SE_mask = SE_masks.pop(-1)['segmentation'] 
        first_seed = swap(findVisualCenter(SE_mask))
        self.SE_Seeds = [swap(findVisualCenter(mask['segmentation'])) for mask in SE_masks]#SE_seeds are saved in the evidence format
        self.cp_mask = SE_mask
        return(first_seed, len(self.SE_Seeds), SE_mask)
    
    def learn(self, input_points, input_labels):
        if input_labels[-1]:
            self.look_for_first_GT_mitochondria = False
            
        evidence, scores, logits = self.predictor.predict(
            point_coords = np.array(input_points),
            point_labels = np.array(input_labels),
            multimask_output = False,
            return_logits = True,
        )
        self.evidence = evidence.squeeze()#image format
        self.cp_mask = self.evidence > self.predictor.model.mask_threshold
        return(self.cp_mask)
    
    def findNewSeed(self):
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getEvidence(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:#Chose most uncertain SE_seed
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : abs(getEvidence(self.evidence, x)))
            new_seed = self.SE_Seeds.pop(0)
        return(new_seed)
        

class FPFNLearner(ActiveLearningSAM):
    """
    Not really an active learner because new seeds are selected using ground truth
    """
    def __init__(self, model, image = None, mask_generator = None, GT_mask = None):
        super().__init__(model, image, mask_generator)
        self.GT_mask = GT_mask
        self.needGroundTruth = True
    
    def setGroundTruthMask(self, mask):
        self.GT_mask = mask.astype("uint8")
    
    def findNewSeed(self):#find new seed using the FP and FN. 
        error = np.bitwise_xor(self.GT_mask, self.cp_mask)
        new_seed = swap(findVisualCenter(error))
        return new_seed

class MaxUncertaintyLearner(ActiveLearningSAM): 
    
    def findNewSeed(self):
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getEvidence(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:#Chose most uncertain seed on the whole image
            cx, cy = np.unravel_index(np.argmax(uncertaintyKL(self.evidence)), self.evidence.shape)
            new_seed = [cy, cx]#swap cx and cy to meet format
        return(new_seed)


class DistTransformLearner(ActiveLearningSAM): 
    
    def __init__(self, model, image = None, mask_generator = None, GT_mask = None):
        super().__init__(model, image, mask_generator)
        p_thresh = 0.8
        self.evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
    
    def findNewSeed(self):
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getEvidence(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:
            uncertainMask = self.evidence < self.evidence_thresh
            new_seed = swap(findVisualCenter(uncertainMask))
        return(new_seed)

class UncertaintyPathLearner(ActiveLearningSAM): 
    
    def __init__(self, model, image = None, mask_generator = None, GT_mask = None):
        super().__init__(model, image, mask_generator)
        p_thresh = 0.95
        self.evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
    
    def findNewSeed(self):
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getEvidence(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:
            new_seed = swap(findNewSeedWithUncertaintyPathDist(self.evidence, self.evidence_thresh))
        return(new_seed)

 