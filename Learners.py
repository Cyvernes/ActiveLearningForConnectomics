import json
import os
import random
import sys
import time
import warnings
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

from filters import *
from next_seeds_strategies import *
from plot_tools import *
from SAM_modifications import SamPredictorWithDropOut
from strategy_selectors import *
from tools import *



class ActiveLearningSAM:
    """This is the fundamental class for the active Learning for Connectomics project.
    This class should be used for every active learner that need not access ground truth.
    
    Please note that this class does not have direct access to Ground Truth data. 
    Instead, it is designed to work in conjunction with other components of the project to incorporate Ground Truth feedback during active learning iterations.
    """
    def __init__(
        self,
        model,
        strategy_selector,
        learning_strategies,
        first_seeds_selector,
        seeds_selection_strategies,
        uncertainty_fn,
        filtering_fn,
        filtering_aux_fn,
        image=None,
        mask_generator=None,
        use_previous_logits=True,
    ) -> None:
        """Constructor of the ActiveLearningSAM class

        :param model: Model used in segmentation (usually it's SAM)
        :type model: _type_
        :param strategy_selector: Function that selects wich strategy will be used
        :type strategy_selector: _type_
        :param learning_strategies: Every learning strategies that will be used during the experiment
        :type learning_strategies: _type_
        :param first_seeds_selector: Function that selects the first points to annotate
        :type first_seeds_selector: _type_
        :param seeds_selection_strategies: Sampling strategies that will be used during the experiment
        :type seeds_selection_strategies: _type_
        :param uncertainty_fn: Function that computes the uncertainty map from the evidence map
        :type uncertainty_fn: _type_
        :param filtering_fn: Function used as a filter during sampling of the next seeds
        :type filtering_fn: _type_
        :param filtering_aux_fn: Auxiliary function that computes the region of interest for the filter
        :type filtering_aux_fn: _type_
        :param image: Image to segment, defaults to None
        :type image: _type_, optional
        :param mask_generator: Automatic mask generator, defaults to None
        :type mask_generator: _type_, optional
        :param use_previous_logits: _description_, defaults to True
        :type use_previous_logits: Parameter for the predictor. If set to true the predictor will use previous prediction during prediction, optional
        """

        self.model = model
        self.image = image
        self.cp_mask = None  # Current predicted mask
        self.need_ground_truth = False
        self.use_previous_logits = use_previous_logits
        self.input_points = []
        self.seed_subsets = []  # Used by FewSeedsForOneMask
        self.label_subsets = []  # Used by FewSeedsForOneMask
        self.input_labels = []
        self.evidence = None
        self.current_strategy_idx = 0
        self.logits = None
        self.a_priori = None
        self.current_strategy_idx = 0
        self.masks_from_single_seeds = None
        self.idx_when_strat_has_changed = []
        self.nb_seed_used = 0
        self.nb_initial_seeds = -1

        self.first_seeds_selector = first_seeds_selector

        self.uncertainty_function = uncertainty_fn

        self.filtering_function = filtering_fn
        self.filtering_aux_function = filtering_aux_fn

        self.strategy_selector = strategy_selector

        self.seeds_selection_strategies = seeds_selection_strategies

        self.count = 0  # debug
        # Instancing a mask generator
        if mask_generator:
            self.mask_generator = mask_generator
        else:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=64,
                points_per_batch=64,
                pred_iou_thresh=0.95,
                stability_score_thresh=0.8,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=0,  # Requires open-cv to run post-processing
            )

        # Instancing a predictor
        self.predictor = SamPredictorWithDropOut(self.model, p=0.2, use_dropout=False)
        self.learning_strategies = [ls(self) for ls in learning_strategies]

    def setData(self, image: np.ndarray) -> None:
        """Load the image data into the learner and initialize attributes.
        
        This method prepares the learner for segmentation by loading the input image data
        and initializing relevant attributes. Upon calling this method, several attributes
        are reinitialized to facilitate the segmentation process.
        
        :param image: Image to segment
        :type image: np.ndarray
        """

        self.image = image
        h, w, _ = image.shape
        self.evidence = np.zeros((h, w), dtype="float32")
        self.predictor.set_image(image)
        self.cp_mask = None
        self.current_strategy_idx = 0

        evidence, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=None,
            multimask_output=False,
            return_logits=True,
        )
        self.a_priori = evidence.squeeze()
        self.masks_from_single_seeds = np.zeros_like(self.a_priori, dtype="int16")
        self.nb_initial_seeds = -1

    def findFirstSeeds(self) -> Tuple[List[Tuple[int, int]], int]:
        """Compute the first points to annotate.    
        
        This method calls the initial seed selection process (first_seeds_selector) usually based on the Segment Everything (SE)
        approach. It generates a list of first seeds for annotation and computes the number of seeds
        available for annotation in the SE_Seeds list.
    
        :return: A tuple containing the list of first seeds to annotate and the total number of seeds available in the SE_Seeds list.
        :rtype: Tuple[List[Tuple[int, int]], int]
        """
        self.SE_masks = sorted(
            self.mask_generator.generate(self.image),
            key=lambda mask: mask["predicted_iou"],
        )  # masks from Segment Everything are in the same format as the image
        self.SE_Seeds = removeTooCloseSeeds(
            [swap(findVisualCenter(mask["segmentation"])) for mask in self.SE_masks]
        )  # SE_seeds are saved in the input format
        nb_seeds = len(self.SE_Seeds)
        self.nb_initial_seeds = nb_seeds
        first_seeds, SE_mask = self.first_seeds_selector(self)
        self.cp_mask = SE_mask
        del self.SE_masks
        return (first_seeds, nb_seeds)

    def learn(self, input_points: list, input_labels: list) -> np.ndarray:
        """Implements an iteration of active learning to improve segmentation.
        
        This method represents a single iteration of the active learning process, where new seeds
        (input_points) and their corresponding labels (input_labels) are used to enhance the
        segmentation. The evidence and uncertainty map are updated according to the selected learning strategy.
        The learning strategy is selected by the strategy_selector that computes the current strategy index (current_strategy_idx).
        The corresponding strategy, modulo the number of strategies,in learning_strategies is used.

        :param input_points: Input seeds
        :type input_points: list
        :param input_labels: Labels corresponding to input seeds
        :type input_labels: list
        :return: Current predicted mask
        :rtype: np.ndarray
        """
        self.input_points = input_points
        self.input_labels = input_labels
        old_strat = self.current_strategy_idx
        self.current_strategy_idx = self.strategy_selector(self)
        if old_strat != self.current_strategy_idx:
            self.idx_when_strat_has_changed.append(len(input_points))
            
        # apply the selected learning strategy
        self.learning_strategies[self.current_strategy_idx % len(self.learning_strategies)]()  
        self.nb_seed_used = len(self.input_points)
        return self.cp_mask

    def findNextSeeds(self) -> list:
        """Uses the evidence map and uncertainty map to sample the next seed for annotation.

        :return: The next seeds (points) to annotate based on the selected strategy.
        :rtype: list
        """

        self.current_strategy_idx = self.strategy_selector(self)
        old_strat = self.current_strategy_idx
        if old_strat != self.current_strategy_idx:
            self.idx_when_strat_has_changed.append(len(self.input_points))
        next_seeds = self.seeds_selection_strategies[
            self.current_strategy_idx % len(self.seeds_selection_strategies)
        ](self)
        return next_seeds


class FPFNLearner(ActiveLearningSAM):
    """This class is not an active learner. Seeds are selected accoring to the most arrorous point.
    """
    def __init__(
        self,
        model,
        strategy_selector,
        learning_strategies,
        seeds_selection_strategies,
        uncertainty_fn,
        filtering_fn,
        filtering_aux_fn,
        image=None,
        mask_generator=None,
        use_previous_logits=True,
    ):
        """Constructor of the FPFNLearner class

        :param model: Model used in segmentation (usually it's SAM)
        :type model: _type_
        :param strategy_selector: Function that selects wich strategy will be used
        :type strategy_selector: _type_
        :param learning_strategies: Every learning strategies that will be used during the experiment
        :type learning_strategies: _type_
        :param first_seeds_selector: Function that selects the first points to annotate
        :type first_seeds_selector: _type_
        :param seeds_selection_strategies: Sampling strategies that will be used during the experiment
        :type seeds_selection_strategies: _type_
        :param uncertainty_fn: Function that computes the uncertainty map from the evidence map
        :type uncertainty_fn: _type_
        :param filtering_fn: Function used as a filter during sampling of the next seeds
        :type filtering_fn: _type_
        :param filtering_aux_fn: Auxiliary function that computes the region of interest for the filter
        :type filtering_aux_fn: _type_
        :param image: Image to segment, defaults to None
        :type image: _type_, optional
        :param mask_generator: Automatic mask generator, defaults to None
        :type mask_generator: _type_, optional
        :param use_previous_logits: Parameter for the predictor. If set to true the predictor will use previous prediction during prediction, defaults to True
        :type use_previous_logits: bool, optional
        """
        super().__init__(
            model,
            strategy_selector,
            learning_strategies,
            seeds_selection_strategies,
            uncertainty_fn,
            filtering_fn,
            filtering_aux_fn,
            image,
            mask_generator,
            use_previous_logits,
        )
        self.GT_mask = None
        self.need_ground_truth = True
    
    def setGroundTruthMask(self, mask: np.ndarray):
        """Sets the ground truth mask

        :param mask: Ground truth mask
        :type mask: np.ndarray
        """
        self.GT_mask = mask.astype("uint8")

    def findNextSeeds(self) -> list:
        """Samples next seed using the false positive pixels and the false negative pixels.

        :return: Next seed to annotate (in a list)
        :rtype: list
        """
        error = np.bitwise_xor(self.GT_mask, self.cp_mask)
        new_seed = swap(findVisualCenter(error))
        return [new_seed]
 

class RandomLearner(ActiveLearningSAM):
    """This class is not an active learner, next seeds are chosen at random
    """
    def __init__(
        self,
        model,
        strategy_selector,
        learning_strategies,
        seeds_selection_strategies,
        uncertainty_fn,
        filtering_fn,
        filtering_aux_fn,
        image=None,
        mask_generator=None,
        use_previous_logits=True,
    ): 
        """Constructor of the RandomLearner class
        Sampling is done at random

        :param model: Model used in segmentation (usually it's SAM)
        :type model: _type_
        :param strategy_selector: Function that selects wich strategy will be used
        :type strategy_selector: _type_
        :param learning_strategies: Every learning strategies that will be used during the experiment
        :type learning_strategies: _type_
        :param first_seeds_selector: Function that selects the first points to annotate
        :type first_seeds_selector: _type_
        :param seeds_selection_strategies: Sampling strategies that will be used during the experiment
        :type seeds_selection_strategies: _type_
        :param uncertainty_fn: Function that computes the uncertainty map from the evidence map
        :type uncertainty_fn: _type_
        :param filtering_fn: Function used as a filter during sampling of the next seeds
        :type filtering_fn: _type_
        :param filtering_aux_fn: Auxiliary function that computes the region of interest for the filter
        :type filtering_aux_fn: _type_
        :param image: Image to segment, defaults to None
        :type image: _type_, optional
        :param mask_generator: Automatic mask generator, defaults to None
        :type mask_generator: _type_, optional
        :param use_previous_logits: Parameter for the predictor. If set to true the predictor will use previous prediction during prediction, defaults to True
        :type use_previous_logits: bool, optional
        """
        super().__init__(
            model,
            strategy_selector,
            learning_strategies,
            seeds_selection_strategies,
            uncertainty_fn,
            filtering_fn,
            filtering_aux_fn,
            image,
            mask_generator,
            use_previous_logits,
        )
        random.seed(0)  # select seed for pseudo random generator

    def findNextSeeds(self) -> np.ndarray:
        """Samples next seed using the false positive pixels and the false negative pixels.

        :return: Next seed to annotate (in a list)
        :rtype: list
        """
        h, w, _ = self.image.shape
        new_seed = [random.randint(0, h - 1), random.randint(0, w - 1)]
        return [new_seed]


class PseudoActiveLearningSAM(ActiveLearningSAM):
    """This class is an active Learner but it can acces ground truth. 
    This allows to simulate specific settings that need to use the ground truth.
    For instance, giving all foreground points at the beginning can only be done in this class.
    """
    def __init__(
        self,
        model,
        strategy_selector,
        learning_strategies,
        first_seeds_selector,
        seeds_selection_strategies,
        uncertainty_fn,
        filtering_fn,
        filtering_aux_fn,
        image=None,
        mask_generator=None,
        use_previous_logits=True,
    ) -> None:
        """Constructor of the ActiveLearningSAM class

        :param model: Model used in segmentation (usually it's SAM)
        :type model: _type_
        :param strategy_selector: Function that selects wich strategy will be used
        :type strategy_selector: _type_
        :param learning_strategies: Every learning strategies that will be used during the experiment
        :type learning_strategies: _type_
        :param first_seeds_selector: Function that selects the first points to annotate
        :type first_seeds_selector: _type_
        :param seeds_selection_strategies: Sampling strategies that will be used during the experiment
        :type seeds_selection_strategies: _type_
        :param uncertainty_fn: Function that computes the uncertainty map from the evidence map
        :type uncertainty_fn: _type_
        :param filtering_fn: Function used as a filter during sampling of the next seeds
        :type filtering_fn: _type_
        :param filtering_aux_fn: Auxiliary function that computes the region of interest for the filter
        :type filtering_aux_fn: _type_
        :param image: Image to segment, defaults to None
        :type image: _type_, optional
        :param mask_generator: Automatic mask generator, defaults to None
        :type mask_generator: _type_, optional
        :param use_previous_logits: _description_, defaults to True
        :type use_previous_logits: Parameter for the predictor. If set to true the predictor will use previous prediction during prediction, optional
        """
        super().__init__(
            model,
            strategy_selector,
            learning_strategies,
            first_seeds_selector,
            seeds_selection_strategies,
            uncertainty_fn,
            filtering_fn,
            filtering_aux_fn,
            image,
            mask_generator,
            use_previous_logits,
        )
        self.GT_mask = None
        self.need_ground_truth = True

    def setGroundTruthMask(self, mask: np.ndarray) -> None:
        """Set the ground truth mask

        :param mask: Ground truth mask
        :type mask: np.ndarray
        """
        self.GT_mask = mask.astype("uint8")
