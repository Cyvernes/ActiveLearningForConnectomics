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
from next_seeds_strategies import *
from SAM_modifications import SamPredictorWithDropOut
from typing import List, Tuple


class ActiveLearningSAM:
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
    ):
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
        self.learning_strategies = learning_strategies
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

        self.count = 0 # debug
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
        self.predictor = predictor = SamPredictorWithDropOut(self.model, p=0.2, use_dropout=False)

    def setData(self, image: np.ndarray) -> None:
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

    def findFirstSeeds(
        self,
    ) -> Tuple[
        List[Tuple[int, int]], int, np.ndarray
    ]:  # masks from Segment Everything are in the same format as the image
        self.SE_masks = sorted(
            self.mask_generator.generate(self.image),
            key=lambda mask: mask["predicted_iou"],
        )  # masks from segement every thing
        self.SE_Seeds = removeTooCloseSeeds(
            [swap(findVisualCenter(mask["segmentation"])) for mask in self.SE_masks]
        )  # SE_seeds are saved in the input format
        nb_seeds = len(self.SE_Seeds)
        self.nb_initial_seeds = nb_seeds
        first_seeds, SE_mask = self.first_seeds_selector(self)
        self.cp_mask = SE_mask
        del self.SE_masks
        return (first_seeds, nb_seeds, SE_mask)

    def learn(self, input_points: list, input_labels: list) -> np.ndarray:
        self.input_points = input_points
        self.input_labels = input_labels
        old_strat = self.current_strategy_idx
        self.current_strategy_idx = self.strategy_selector(self)
        if old_strat != self.current_strategy_idx:
            self.idx_when_strat_has_changed.append(len(input_points))
        self.learning_strategies[self.current_strategy_idx % len(self.learning_strategies)](self)
        self.nb_seed_used = len(self.input_points)
        return self.cp_mask

    def findNextSeeds(self) -> list:
        self.current_strategy_idx = self.strategy_selector(self)
        old_strat = self.current_strategy_idx
        if old_strat != self.current_strategy_idx:
            self.idx_when_strat_has_changed.append(len(self.input_points))
        next_seeds = self.seeds_selection_strategies[self.current_strategy_idx % len(self.seeds_selection_strategies)](
            self
        )
        return next_seeds


class FPFNLearner(ActiveLearningSAM):
    """
    Not really an active learner because next seeds are selected using ground truth
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
        self.GT_mask = mask.astype("uint8")

    def findNewSeed(self) -> np.ndarray:  # find next seed using the FP and FN.
        error = np.bitwise_xor(self.GT_mask, self.cp_mask)
        new_seed = swap(findVisualCenter(error))
        return [new_seed]


class RandomLearner(ActiveLearningSAM):
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

    def findNewSeed(self) -> np.ndarray:
        h, w, _ = self.image.shape
        new_seed = [random.randint(0, h - 1), random.randint(0, w - 1)]
        return [new_seed]


class PseudoActiveLearningSAM(ActiveLearningSAM):
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
    ):
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

    def setGroundTruthMask(self, mask: np.ndarray):
        self.GT_mask = mask.astype("uint8")
