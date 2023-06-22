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
import warnings
from tools import *
from plot_tools import *
from filters import *

class ActiveLearningSAM:
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        self.model = model
        self.image = image
        self.cp_mask = None #Current predicted mask
        self.need_ground_truth = False
        self.use_previous_logits = use_previous_logits
        
        #selecting the uncertainty function
        if uncertainty_fn_type == "H":
            self.uncertainty_function = uncertaintyH
        elif uncertainty_fn_type == "KL":
            self.uncertainty_function = uncertaintyKL
        else:
            raise ValueError('Unknown uncertainty function type')
        
        #selecting the filtering function (it is used to filter the uncertainty map before finding the next seed)
        if   filtering_fn == "None":
            self.filtering_function = filterTrivial
        elif filtering_fn == "Dist":
            self.filtering_function = filterWithDist
        elif filtering_fn == "DistWithBorder":
            self.filtering_function = filterWithDistWithBorder
        elif filtering_fn == "DistSkeleton":
            self.filtering_function = filterWithDistSkeleton
        elif filtering_fn == "Percentile":
            self.filtering_function = filterWithPercentile
        else:
            raise ValueError('Unknown filtering function type')
            
        #Instancing a mask generator
        if mask_generator:
            self.mask_generator = mask_generator
        else: 
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
        self.logits = None
    
    def setData(self, image : np.ndarray) -> None:
        self.image = image
        self.predictor.set_image(image)
        self.cp_mask = None
    
    def findFirstSeed(self) -> Tuple[int, int]:#masks from Segment Everything are in the same format as the image
        SE_masks = sorted(self.mask_generator.generate(self.image), key = lambda mask: mask['predicted_iou'])#masks from segement every thing
        nb_seeds = len(SE_masks)
        SE_mask = SE_masks.pop(-1)['segmentation'] 
        first_seed = swap(findVisualCenter(SE_mask))
        self.SE_Seeds = [swap(findVisualCenter(mask['segmentation'])) for mask in SE_masks]#SE_seeds are saved in the evidence format
        self.cp_mask = SE_mask
        return(first_seed, nb_seeds, SE_mask)
    
    def learn(self, input_points : np.ndarray, input_labels : np.ndarray) -> np.ndarray:
        if input_labels[-1]:
            self.look_for_first_GT_mitochondria = False
            
        evidence, scores, self.logits = self.predictor.predict(
            point_coords = np.array(input_points),
            point_labels = np.array(input_labels),
            mask_input = self.logits if self.use_previous_logits else None,
            multimask_output = False,
            return_logits = True,
        )
        self.evidence = evidence.squeeze()#image format
        self.cp_mask = self.evidence > self.predictor.model.mask_threshold
        return(self.cp_mask)
    
    def findNewSeed(self) -> np.ndarray:
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getValueinArrFromInputFormat(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:#Chose most uncertain SE_seed
            uncertainty = self.uncertainty_function(self.evidence)
            uncertainty = self.filtering_function(uncertainty, self.evidence)
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : abs(getValueinArrFromInputFormat(self.evidence, x)))
            new_seed = self.SE_Seeds.pop(0)
        return(new_seed)
        

class FPFNLearner(ActiveLearningSAM):
    """
    Not really an active learner because new seeds are selected using ground truth
    """
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, uncertainty_fn_type, filtering_fn, image, mask_generator)
        self.GT_mask = GT_mask
        self.need_ground_truth = True
        if filtering_fn != "None":
            warnings.warn("This learner does not need a filtering function.")
    
    def setGroundTruthMask(self, mask : np.ndarray):
        self.GT_mask = mask.astype("uint8")
    
    def findNewSeed(self) -> np.ndarray:#find new seed using the FP and FN. 
        error = np.bitwise_xor(self.GT_mask, self.cp_mask)
        new_seed = swap(findVisualCenter(error))
        return new_seed


class DistTransformLearner(ActiveLearningSAM): 
    
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, uncertainty_fn_type, filtering_fn, image, mask_generator)
        p_thresh = 0.8
        self.evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
        if filtering_fn != "None":
            warnings.warn("This learner does not need a filtering function.")
    
    def findNewSeed(self) -> np.ndarray:
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getValueinArrFromInputFormat(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:
            uncertainMask = self.evidence < self.evidence_thresh
            new_seed = swap(findVisualCenter(uncertainMask))
        return(new_seed)


class RandomLearner(ActiveLearningSAM):
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, uncertainty_fn_type, filtering_fn, image, mask_generator, use_previous_logits)
        random.seed(0)
        if filtering_fn != "None":
            warnings.warn("This learner does not need a filtering function.")

    
    def findNewSeed(self) -> np.ndarray:
        h, w, _ = self.image.shape
        new_seed = [random.randint(0, h - 1), random.randint(0, w - 1)]
        return(new_seed)
        

class UncertaintyPathLearner(ActiveLearningSAM): 
    
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, uncertainty_fn_type, filtering_fn, image, mask_generator)
        p_thresh = 0.95
        self.evidence_thresh = np.log(p_thresh/ (1 - p_thresh))
    
    def findNewSeed(self) -> np.ndarray:
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getValueinArrFromInputFormat(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:
            uncertainty = self.uncertainty_function(self.evidence)
            uncertainty = self.filtering_function(uncertainty, self.evidence)
            thresh = self.uncertainty_function(self.evidence_thresh)
            distances = UncertaintyPathDist(uncertainty, self.evidence, thresh)
            cx, cy = np.unravel_index(np.argmax(distances), distances.shape)
            new_seed = [cy, cx]
        return(new_seed)


class MaxUncertaintyLearner(ActiveLearningSAM): 
    
    def __init__(self, model, uncertainty_fn_type="H", filtering_fn="None", image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, uncertainty_fn_type, filtering_fn, image, mask_generator)

    def findNewSeed(self) -> np.ndarray:
        if self.look_for_first_GT_mitochondria:#Chose seed with most evidence to find a mitochondrion
            self.SE_Seeds = sorted(self.SE_Seeds, key = lambda x : getValueinArrFromInputFormat(self.evidence, x))
            new_seed = self.SE_Seeds.pop(-1)
        else:#Chose most uncertain seed 
            uncertainty = self.uncertainty_function(self.evidence)
            uncertainty = self.filtering_function(uncertainty, self.evidence)
            cx, cy = np.unravel_index(np.argmax(uncertainty), self.evidence.shape)
            new_seed = [cy, cx]#swap cx and cy to meet format
        return(new_seed)