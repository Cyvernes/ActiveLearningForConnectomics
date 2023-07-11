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
from Learners import *
from tools import *
from plot_tools import *
from filters import *
from learning_strategies import *

SUBSET_SIZE = 1
TRAIN_RATIO = 1
LOAD_DATA_ONCE_FOR_ALL = True
CHOOSE_DATA_AT_RANDOM = False
LOAD_ALL_DATA = True

LEARNER_TYPE = "Active Learning"  # {"Active Learning", "FPFN", "Random"}
STRATEGY_SELECTOR = changeAfterAGivenAmountOfSeed  # {singleStrat, changeAtFirstMito, changeAfterAGivenAmountOfSeed}
LEARNING_STRATEGIES = [oneSeedForOneMaskLS]  # {basicLS, oneSeedForOneMaskLS}
USE_PREVIOUS_LOGITS = False  # change how Learning strategies use previous logits (only change basicLS now) (may be deprecated in the future)
SEED_SELECTION_STRATEGIES = [
    ArgmaxEvInSESeeds,
    ArgmaxUncertainty,
]  # {ArgmaxEvInSESeeds, ArgmaxDist, ArgmaxUncertainty, ArgmaxUncertaintyPathDist, ArgmaxUncertaintyInSESeeds}
UNCERTAINTY_FUNCTION_TYPE = uncertaintyH  # {uncertaintyH, uncertaintyKL}
FILTERING_FUNCTION = filterWithDist  # {filterTrivial, filterWithDist, filterWithDistWithBorder, filterWithPercentile, filterWithDistSkeleton}
FILTERING_AUX_FONCTION = (
    evidenceSmallerOrEqualToZero  # {evidenceSmallerOrEqualToZero, threshOnUncertainty}
)
USE_BUDGET = False
ANNOTATION_BUDGET = 20

SAVE_INTERMEDIATE_RESULTS = False
SAVE_IMAGE_WITH_GT = False
SAVE_FIRST_SEED = False
SAVE_FINAL_IOU_EVOLUTION = False
SAVE_UNCERTAINTY_PERCENTILES = False

FILE_WITH_ALL_LINKS = (
    "/n/home12/cyvernes/working_directory/cem_mitolab_dataset_links.json"
)
FOLDER_FOR_INTERMEDIATE_RESULTS = "working_directory/results/intermediate results/"
FOLDER_FOR_FINAL_RESULTS = "working_directory/results/final results/"
SPECIFIC_IMAGE_LINKS = [
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-R/images/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/jrc_ctl-id8-4_openorganelle/images/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/images/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-H/images/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/52f72Bc125o7v94Ep850_2D/images/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff",
]
SPECIFIC_MASK_LINKS = [
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-R/masks/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/jrc_ctl-id8-4_openorganelle/masks/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/masks/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-H/masks/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff",
    "/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/52f72Bc125o7v94Ep850_2D/masks/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff",
]
"""
FOLDER_FOR_INTERMEDIATE_RESULTS = "./temp/"
FOLDER_FOR_FINAL_RESULTS = "./final results/"
SPECIFIC_IMAGE_LINKS = [
    "../../data/cem_mitolab/Wei2020_MitoEM-R/images/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff",
    "../../data/cem_mitolab/jrc_ctl-id8-4_openorganelle/images/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff",
    "../../data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/images/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff",
    "../../data/cem_mitolab/Wei2020_MitoEM-H/images/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff",
    "../../data/cem_mitolab/52f72Bc125o7v94Ep850_2D/images/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff",
]
SPECIFIC_MASK_LINKS = [
    "../../data/cem_mitolab/Wei2020_MitoEM-R/masks/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff",
    "../../data/cem_mitolab/jrc_ctl-id8-4_openorganelle/masks/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff",
    "../../data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/masks/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff",
    "../../data/cem_mitolab/Wei2020_MitoEM-H/masks/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff",
    "../../data/cem_mitolab/52f72Bc125o7v94Ep850_2D/masks/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff",
]
"""
SPECIFIC_IMAGE_LINKS = [SPECIFIC_IMAGE_LINKS[3]]
SPECIFIC_MASK_LINKS = [SPECIFIC_MASK_LINKS[3]]

if __name__ == "__main__":
    """
    Check the environment and the cuda device
    """
    print("-------------------------------------")
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Loading data
    """
    print("-------------------------------------")
    print("Retrieving the data...")
    if CHOOSE_DATA_AT_RANDOM:
        with open(FILE_WITH_ALL_LINKS) as infile:
            dataset_links = json.load(infile)
            print(
                "There is" if len(dataset_links["images"]) == 1 else "There are",
                len(dataset_links["images"]),
                "images available in the dataset.",
            )
        subset_idx = random.sample(range(len(dataset_links["images"])), SUBSET_SIZE)
        images_links = [dataset_links["images"][idx] for idx in subset_idx]
        masks_links = [dataset_links["masks"][idx] for idx in subset_idx]
        del subset_idx
    elif LOAD_ALL_DATA:
        images_links, masks_links = retrieveLinks()
    else:
        images_links = SPECIFIC_IMAGE_LINKS
        masks_links = SPECIFIC_MASK_LINKS

    print(
        "There is" if len(images_links) == 1 else "There are",
        len(images_links),
        "selected images.",
    )

    if LOAD_DATA_ONCE_FOR_ALL:
        print("Loading the data once for all...")
        train_images = [
            cv2.imread(images_links[idx])
            for idx in range(int(TRAIN_RATIO * len(images_links)))
        ]
        test_images = [
            cv2.imread(images_links[idx])
            for idx in range(int(TRAIN_RATIO * len(images_links)), len(images_links))
        ]
        train_masks = [
            np.any(cv2.imread(masks_links[idx]) != [0, 0, 0], axis=-1)
            for idx in range(int(TRAIN_RATIO * len(images_links)))
        ]
        test_masks = [
            np.any(cv2.imread(masks_links[idx]) != [0, 0, 0], axis=-1)
            for idx in range(int(TRAIN_RATIO * len(images_links)), len(images_links))
        ]

    """
    Loading SAM
    """
    print("-------------------------------------")
    print("Loading the model...")

    # Loading model weights
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    """
    Training
    """
    print("-------------------------------------")
    print("Training...")

    # Instancing a learning algorithm
    if LEARNER_TYPE == "Active Learning":
        learner = ActiveLearningSAM(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FONCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    elif LEARNER_TYPE == "FPFN":
        learner = FPFNLearner(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FONCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    elif LEARNER_TYPE == "Random":
        learner = RandomLearner(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FONCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    else:
        raise ValueError("Unknown learner type")

    if LOAD_ALL_DATA:
        final_IoUs = []
        final_FPs = []
        final_FNs = []
        final_percentiles = []
        final_nb_seeds = []

    for idx in range(int(TRAIN_RATIO * len(images_links))):
        print(
            "-------------------------------------",
            idx + 1,
            "/",
            int(TRAIN_RATIO * len(images_links)),
        )
        # Load image and ground truth mask
        if LOAD_DATA_ONCE_FOR_ALL:
            image = train_images[idx]
            GT_mask = train_masks[idx]
        else:
            image = cv2.imread(images_links[idx])
            GT_mask = np.any(cv2.imread(masks_links[idx]) != [0, 0, 0], axis=-1)

        if SAVE_IMAGE_WITH_GT:
            plotAndSaveImageWithGT(FOLDER_FOR_INTERMEDIATE_RESULTS, image, GT_mask, idx)

        learner.setData(image)

        # Give image and GTmask (if needed) to the learner
        if learner.need_ground_truth:
            learner.setGroundTruthMask(GT_mask)

        # Find the first seed to query
        first_seed, nb_seeds, first_mask = learner.findFirstSeed()
        if len(first_seed) == 0:  # avoid empty list
            print("No first seed was given")
            continue

        if SAVE_FIRST_SEED:
            plotAndSaveImageWithFirstSeed(
                FOLDER_FOR_INTERMEDIATE_RESULTS, image, first_mask, first_seed, idx
            )

        # Main loop

        if SAVE_UNCERTAINTY_PERCENTILES:
            percentiles_points = list(range(0, 100, 5))
            percentiles = [[] for i in percentiles_points]

        input_points = []
        input_labels = []
        IoUs = []
        FPs = []
        FNs = []
        next_seeds = [first_seed.copy()]
        NBs = []

        if USE_BUDGET:
            nb_seeds = ANNOTATION_BUDGET

        for i in range(nb_seeds):
            input_points += next_seeds
            input_labels += [getLabel(new_seed, GT_mask) for new_seed in next_seeds]

            learner.learn(input_points, input_labels)

            # Save results
            NBs.append(NBs[-1] + len(next_seeds) if len(NBs) > 0 else len(next_seeds))
            IoUs.append(IoU(learner.cp_mask, GT_mask))
            FPs.append(FP(learner.cp_mask, GT_mask))
            FNs.append(FN(learner.cp_mask, GT_mask))

            # find the next seeds
            if i != nb_seeds - 1:
                next_seeds = learner.findNextSeeds()

            if SAVE_INTERMEDIATE_RESULTS:  # draw i-th prediction
                plotAndSaveIntermediateResults(
                    FOLDER_FOR_INTERMEDIATE_RESULTS,
                    learner,
                    next_seeds,
                    image,
                    GT_mask,
                    IoUs,
                    FNs,
                    FPs,
                    i,
                    idx,
                    nb_seeds,
                    NBs,
                )

            if SAVE_UNCERTAINTY_PERCENTILES:
                savePercentiles(learner, percentiles_points, percentiles)

        if SAVE_UNCERTAINTY_PERCENTILES:
            savePercentilesPlot(FOLDER_FOR_FINAL_RESULTS, NBs, percentiles, idx)

        if SAVE_FINAL_IOU_EVOLUTION:
            plotAndSaveFinalIoUEvolution(FOLDER_FOR_FINAL_RESULTS, NBs, IoUs, idx)

        if LOAD_ALL_DATA:
            final_IoUs.append(max(IoUs))
            final_FPs.append(max(FPs))
            final_FNs.append(max(FNs))
            final_nb_seeds.append(nb_seeds)
            # final_percentiles.append(max(percentiles))

        """
        Testing
        """
        print("-------------------------------------")
        print("Testing...")

    if LOAD_ALL_DATA:
        plotAllData(
            FOLDER_FOR_FINAL_RESULTS,
            final_IoUs,
            final_FPs,
            final_FNs,
            final_nb_seeds,
        )
