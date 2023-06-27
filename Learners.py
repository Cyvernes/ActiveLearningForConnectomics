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
from strategy_selectors import *
from new_seeds_strategies import *


class ActiveLearningSAM:
    def __init__(self, model, strategy_selector, seeds_selection_strategies, uncertainty_fn, filtering_fn, image=None, mask_generator=None, use_previous_logits=True):
        self.model = model
        self.image = image
        self.cp_mask = None #Current predicted mask
        self.need_ground_truth = False
        self.use_previous_logits = use_previous_logits
        self.input_points = []
        self.input_labels = []
        self.evidence = None
        self.strategy_idx = 0
        self.logits = None
        
        #selecting the uncertainty function
        self.uncertainty_function = uncertainty_fn
        
        #selecting the filtering function (it is used to filter the uncertainty map before finding the next seed)
        self.filtering_function = filtering_fn

        #selecting the strategy selector
        self.strategy_selector = strategy_selector
        
        #selecting the strategies
        self.seeds_selection_strategies = seeds_selection_strategies
        
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

    
    def setData(self, image : np.ndarray) -> None:
        self.image = image
        self.predictor.set_image(image)
        self.cp_mask = None
    
    def findFirstSeed(self) -> Tuple[Tuple[int, int], int, np.ndarray]:#masks from Segment Everything are in the same format as the image
        SE_masks = sorted(self.mask_generator.generate(self.image), key = lambda mask: mask['predicted_iou'])#masks from segement every thing
        nb_seeds = len(SE_masks)
        SE_mask = SE_masks.pop(-1)['segmentation'] 
        first_seed = swap(findVisualCenter(SE_mask))
        self.SE_Seeds = [swap(findVisualCenter(mask['segmentation'])) for mask in SE_masks]#SE_seeds are saved in the input format
        self.cp_mask = SE_mask
        return(first_seed, nb_seeds, SE_mask)
    
    def learn(self, input_points : list, input_labels : list) -> np.ndarray:
        self.input_points = input_points
        self.input_labels = input_labels
            
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
    
    def findNewSeeds(self) -> list:
        self.strategy_idx = self.strategy_selector(int(self.strategy_idx), self.input_points, self.input_labels)
        new_seeds = self.seeds_selection_strategies[self.strategy_idx](self)
        return(new_seeds)

class FPFNLearner(ActiveLearningSAM):
    """
    Not really an active learner because new seeds are selected using ground truth
    """
    def __init__(self, model, strategy_selector, seeds_selection_strategies, uncertainty_fn=uncertaintyH, filtering_fn=filterTrivial, image=None, mask_generator=None, use_previous_logits=True, GT_mask = None):
        super().__init__(model, strategy_selector, seeds_selection_strategies, uncertainty_fn, filtering_fn, image, mask_generator, use_previous_logits)
        self.GT_mask = GT_mask
        self.need_ground_truth = True
    
    def setGroundTruthMask(self, mask : np.ndarray):
        self.GT_mask = mask.astype("uint8")
    
    def findNewSeed(self) -> np.ndarray:#find new seed using the FP and FN. 
        error = np.bitwise_xor(self.GT_mask, self.cp_mask)
        new_seed = swap(findVisualCenter(error))
        return([new_seed])

class RandomLearner(ActiveLearningSAM):
    def __init__(self, model, strategy_selector, seeds_selection_strategies, uncertainty_fn=uncertaintyH, filtering_fn=filterTrivial, image=None, mask_generator=None, use_previous_logits=True):
        super().__init__(model, strategy_selector, seeds_selection_strategies, uncertainty_fn, filtering_fn, image, mask_generator, use_previous_logits)
        random.seed(0)

    def findNewSeed(self) -> np.ndarray:
        h, w, _ = self.image.shape
        new_seed = [random.randint(0, h - 1), random.randint(0, w - 1)]
        return([new_seed])