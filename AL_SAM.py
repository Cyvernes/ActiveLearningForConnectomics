""" The purpose of this script is to facilitate learner testing on individual images. 
By utilizing this script, users can efficiently evaluate the performance of their learners or models on specific input images.
"""
import torch
import torchvision
import numpy as np
import cv2
import sys
import json
import time
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import random
import matplotlib.pyplot as plt
from Learners import *
from tools import *
from plot_tools import *
from filters import *
from learning_strategies import *
from first_seeds_selector import *
from data_tools import *

####################################################################
#                                                                  #
#                   MODIFY CONFIGURATION HERE                      #
#                                                                  #
####################################################################

# Learner parameters
LEARNER_TYPE = "Pseudo Active Learning"  # {"Active Learning", "Pseudo Active Learning", "FPFN", "Random"}
STRATEGY_SELECTOR = singleStrat  # {singleStrat, changeAtFirstMito, changeAfterASpecificNumberOfSeed, changeAfterASpecificNumberOfSeenMito}
LEARNING_STRATEGIES = [
    FewSeedsForOneMaskLS
]  # {BasicLS, OneSeedForOneMaskLS, OneSeedForOneMaskLSWithDropOut, FewSeedsForOneMaskLS}
FIRST_SEEDS_SELECTOR = aGivenAmountOfForegroundSeeds  # {popLastSESeeds, allSESeeds, allForegroundSESeeds /!\ need Pseudo Active Learning, aGivenAmountOfForegroundSESeeds}
SEED_SELECTION_STRATEGIES = [
    ArgmaxUncertainty
]  # {ArgmaxEvInSESeeds, ArgmaxUncertainty, ArgmaxUncertaintyPathDist, ArgmaxUncertaintyInSESeeds, ArgmaxEvidence, ArgmaxForegroundProbability}
UNCERTAINTY_FUNCTION_TYPE = uncertaintyH  # {uncertaintyH, uncertaintyKL}
FILTERING_FUNCTION = HybridGDFKS_Dist  # {filterTrivial, filterWithDist, filterWithDistWithBorder, filterWithPercentile, filterWithDistSkeleton, hardFilter, filterGaussianDistFromKnownSeeds, HybridGDFKS_hard, HybridGDFKS_Dist}
FILTERING_AUX_FUNCTION = NotInMasksFromSegmentationStrategy  # {evidenceSmallerOrEqualToZero, threshOnUncertainty, NotInMasksFromSegmentationStrategy}
USE_PREVIOUS_LOGITS = False  # change how Learning strategies use previous logits (only change basicLS now) (may be deprecated in the future)

# SAM checkpoint
SAM_CHECKPOINT = "/n/home12/cyvernes/working_directory/SAM_checkpoints/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

# Budget parameters
USE_BUDGET = True
ANNOTATION_BUDGET = 20

# Plots and results parameters
SAVE_INTERMEDIATE_RESULTS = True
SAVE_IMAGE_WITH_GT = True
SAVE_FIRST_SEED = True
SAVE_FINAL_IOU_EVOLUTION = True
SAVE_UNCERTAINTY_PERCENTILES = True
SAVE_AGGREGATED_RESULTS = False

# Data parameters
SUBSET_SIZE = 1 # Number of images to select in the dataset
TRAIN_RATIO = 1 # Allways set to 1 if there is no testing procedure(currently there is no training on multiple images so no testing)
LOAD_DATA_ONCE_FOR_ALL = True # Load all images on the memory to reduce loading time
CHOOSE_DATA_AT_RANDOM = False # If set to True SUBSET_SIZE images are chosen from the dataset, otherwise images are selected in SPECIFIC_IMAGE_LINKS
LOAD_ONE_IMAGE_IN_EACH_FOLDER = False # If the dataset is splitted on different folders
FILE_WITH_ALL_LINKS = "/n/home12/cyvernes/working_directory/Kidney_HE_glom_capsule_dataset_links.json" # JSON file that contains every link of every image on the dataset.
FOLDER_FOR_INTERMEDIATE_RESULTS = "working_directory/results/intermediate results/" # Folder in which intermediate results are saved.
FOLDER_FOR_FINAL_RESULTS = "working_directory/results/final results/" # Folder in which final results are saved.
SPECIFIC_IMAGE_LINKS = [
    "/n/home12/cyvernes/Kidney/pas-gcapsule-data/im_10.png",
] # if CHOOSE_DATA_AT_RANDOM is set to False, images are loaded from these links.
SPECIFIC_MASK_LINKS = [
    "/n/home12/cyvernes/Kidney/pas-gcapsule-data/im_10_mask_capsule.png",
]  # if CHOOSE_DATA_AT_RANDOM is set to False, masks are loaded from these links.


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
    elif LOAD_ONE_IMAGE_IN_EACH_FOLDER:
        images_links, masks_links = retrieveLinksForOneImageInEachFolder()
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
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
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
            FIRST_SEEDS_SELECTOR,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FUNCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    elif LEARNER_TYPE == "Pseudo Active Learning":
        learner = PseudoActiveLearningSAM(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            FIRST_SEEDS_SELECTOR,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FUNCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    elif LEARNER_TYPE == "FPFN":
        learner = FPFNLearner(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            FIRST_SEEDS_SELECTOR,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FUNCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    elif LEARNER_TYPE == "Random":
        learner = RandomLearner(
            sam,
            STRATEGY_SELECTOR,
            LEARNING_STRATEGIES,
            FIRST_SEEDS_SELECTOR,
            SEED_SELECTION_STRATEGIES,
            UNCERTAINTY_FUNCTION_TYPE,
            FILTERING_FUNCTION,
            FILTERING_AUX_FUNCTION,
            use_previous_logits=USE_PREVIOUS_LOGITS,
        )
    else:
        raise ValueError("Unknown learner type")

    if SAVE_AGGREGATED_RESULTS:
        Warning.warns("Variable names may not be the good ones")
        images_max_IoUs = []
        images_FPs_at_max_IoU = []
        images_FNs_at_max_IoU = []
        images_percentiles = []
        images_nb_seeds = []

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
        first_seeds, nb_seeds = learner.findFirstSeeds()
        if len(first_seeds) == 0:  # avoid empty list
            print("No first seed was given")
            continue

        if SAVE_FIRST_SEED:
            plotAndSaveImageWithFirstSeed(
                FOLDER_FOR_INTERMEDIATE_RESULTS, image, first_seeds, idx
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
        next_seeds = first_seeds.copy()
        NBs = []

        budget = ANNOTATION_BUDGET if USE_BUDGET else nb_seeds
        nb_annotations = 0
        count = 0
        while nb_annotations < budget:
            print("Computing result n°", count + 1,"---------")
            learner.count = count + 1
            input_points += next_seeds
            input_labels += [getLabel(new_seed, GT_mask) for new_seed in next_seeds]
            nb_annotations += len(next_seeds)
            learner.learn(input_points, input_labels)
            # Save results
            NBs.append(len(input_points))
            IoUs.append(IoU(learner.cp_mask, GT_mask))
            FPs.append(FP(learner.cp_mask, GT_mask))
            FNs.append(FN(learner.cp_mask, GT_mask))

            # find the next seeds
            if nb_annotations < budget:
                next_seeds = learner.findNextSeeds()
            else:
                next_seeds = []

            count += 1
            print("Result n°", count, "has been computed")
            if SAVE_INTERMEDIATE_RESULTS:  # draw count-th prediction
                plotAndSaveIntermediateResults(
                    FOLDER_FOR_INTERMEDIATE_RESULTS,
                    learner,
                    next_seeds,
                    image,
                    GT_mask,
                    IoUs,
                    FNs,
                    FPs,
                    count,
                    idx,
                    NBs,
                )

            if SAVE_UNCERTAINTY_PERCENTILES:
                savePercentiles(learner, percentiles_points, percentiles)

        max_IoU = max(IoUs)
        for i, iou in enumerate(IoUs):
            if iou >= 0.9 * max_IoU:
                print(f"Used {NBs[i]} annotations to reach 90% of max IoU")
                break
        if SAVE_UNCERTAINTY_PERCENTILES:
            savePercentilesPlot(FOLDER_FOR_FINAL_RESULTS, NBs, percentiles, idx)

        if SAVE_FINAL_IOU_EVOLUTION:
            plotAndSaveFinalIoUEvolution(FOLDER_FOR_FINAL_RESULTS, NBs, IoUs, idx)

        print("test:", max(IoUs))
        if SAVE_AGGREGATED_RESULTS:
            images_max_IoUs.append(max(IoUs))
            images_FPs_at_max_IoU.append(max(FPs))
            images_FNs_at_max_IoU.append(max(FNs))
            images_nb_seeds.append(nb_seeds)
            # images_percentiles.append(max(percentiles))

    if SAVE_AGGREGATED_RESULTS:
        plotandSaveAggregatedResults(
            FOLDER_FOR_FINAL_RESULTS,
            images_max_IoUs,
            images_FPs_at_max_IoU,
            images_FNs_at_max_IoU,
            images_nb_seeds,
        )
        """
        Testing
        """
