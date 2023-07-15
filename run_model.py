import json
from random import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from segment_anything import sam_model_registry

import filters
import first_seeds_selector
import learning_strategies
import new_seeds_strategies
import next_seeds_strategies
import strategy_selectors
import tools
from Learners import *
from data_tools import *
from plot_tools import *
from tools import *

def load_config(base_json_file, setting_json_file):
    with open(base_json_file, 'r') as f:
        base_config = json.load(f)
    with open(setting_json_file, 'r') as f:
        setting_config = json.load(f)
    for key in setting_config:
        if key in base_config and isinstance(base_config[key], dict):
            base_config[key].update(setting_config[key])
        else:
            base_config[key] = setting_config[key]
    return base_config

def check_environment():
    print("-------------------------------------")
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sam(config):
    print("-------------------------------------")
    print("Loading the model...")
    sam_checkpoint = config['MODEL_WEIGHTS_PATH']
    model_type = config["MODEL_TYPE"]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=config["DEVICE"])
    return sam

def setup_learner(config, sam):
    learner_types = {"Active Learning": ActiveLearningSAM, "Pseudo Active Learning": PseudoActiveLearningSAM, "FPFN": FPFNLearner, "Random": RandomLearner}
    try:
        LearnerClass = learner_types[config["LEARNER_TYPE"]]
    except KeyError:
        raise ValueError("Unknown learner type")
    return LearnerClass(sam, getattr(strategy_selectors, config["STRATEGY_SELECTOR"]),
                        [getattr(learning_strategies, strategy) for strategy in config["LEARNING_STRATEGIES"]],
                        getattr(first_seeds_selector, config["FIRST_SEEDS_SELECTOR"]),
                        [getattr(next_seeds_strategies, strategy) for strategy in config["SEED_SELECTION_STRATEGIES"]],
                        getattr(tools, config["UNCERTAINTY_FUNCTION_TYPE"]),
                        getattr(filters, config["FILTERING_FUNCTION"]),
                        getattr(filters, config["FILTERING_AUX_FONCTION"]), 
                        use_previous_logits=config["USE_PREVIOUS_LOGITS"])

def find_first_seeds(learner, image, GT_mask, idx, config):
    
    learner.setData(image)

    if learner.need_ground_truth:
        learner.setGroundTruthMask(GT_mask)
    print(learner)
    first_seeds, nb_seeds, first_mask = learner.findFirstSeeds()

    if len(first_seeds) == 0:
        print("No first seed was given")
        return None

    if config["SAVE_FIRST_SEED"]:
        plotAndSaveImageWithFirstSeed(config["FOLDER_FOR_INTERMEDIATE_RESULTS"], image, first_seeds, idx)
    
    if config["SAVE_IMAGE_WITH_GT"]:
        plotAndSaveImageWithGT(config["FOLDER_FOR_INTERMEDIATE_RESULTS"], image, GT_mask, idx)
    return first_seeds, nb_seeds

def run_learning_process(learner, GT_mask, first_seeds, nb_seeds, image, idx, config):
    input_points = []
    input_labels = []
    IoUs = []
    FPs = []
    FNs = []
    NBs = []
    percentiles_points = list(range(0, 100, 5))
    percentiles = [[] for i in percentiles_points]
    next_seeds = first_seeds.copy()
    budget = config["training_parameters"]["ANNOTATION_BUDGET"] if config["training_parameters"]["USE_BUDGET"] else nb_seeds
    nb_annotations = 0
    count = 0
    while nb_annotations < budget:
        input_points += next_seeds
        input_labels += [getLabel(new_seed, GT_mask) for new_seed in next_seeds]
        nb_annotations += len(next_seeds)
        learner.learn(input_points, input_labels)
        NBs.append(len(input_points))
        IoUs.append(IoU(learner.cp_mask, GT_mask))
        FPs.append(FP(learner.cp_mask, GT_mask))
        FNs.append(FN(learner.cp_mask, GT_mask))
        if nb_annotations < budget:
            next_seeds = learner.findNextSeeds()
        else:
            next_seeds = []
        count += 1
        if config["plot_parameters"]["SAVE_INTERMEDIATE_RESULTS"]:
            plotAndSaveIntermediateResults(config["plot_parameters"]["FOLDER_FOR_INTERMEDIATE_RESULTS"], learner, next_seeds, image, GT_mask, IoUs, FNs, FPs, count, idx, NBs)
        if nb_annotations == budget-1 and config["plot_parameters"]["SAVE_FINAL_RESULT"]:
            plotAndSaveIntermediateResults(config["plot_parameters"]["FOLDER_FOR_INTERMEDIATE_RESULTS"], learner, next_seeds, image, GT_mask, IoUs, FNs, FPs, count, idx, NBs)
        if config["plot_parameters"]["SAVE_UNCERTAINTY_PERCENTILES"]:
            savePercentiles(learner, percentiles_points, percentiles)

    return NBs, IoUs, FPs, FNs, percentiles

def plots_for_single_image(NBs, IoUs, FPs, FNs, percentiles, idx, config):
    max_IoU = max(IoUs)
    for i, iou in enumerate(IoUs):
        if iou >= 0.9 * max_IoU:
            print(f"Used {i} iterations to reach 90% of max IoU")
            break
    if config["SAVE_UNCERTAINTY_PERCENTILES"]:
        savePercentilesPlot(config["FOLDER_FOR_FINAL_RESULTS"], NBs, percentiles, idx)

    if config["SAVE_FINAL_IOU_EVOLUTION"]:
        plotAndSaveFinalIoUEvolution(config["FOLDER_FOR_FINAL_RESULTS"], NBs, IoUs, idx)


def run_model_single_image(learner, image_link, mask_link, idx, aggregated_results, config):
    image = cv2.imread(image_link)                          
    GT_mask = np.any(cv2.imread(mask_link) != [0, 0, 0], axis=-1)

    first_seeds, nb_seeds = find_first_seeds(learner, image, GT_mask, idx, config["plot_parameters"])
    NBs, IoUs, FPs, FNs, percentiles = run_learning_process(learner, GT_mask, first_seeds, nb_seeds, image, idx, config)
    plots_for_single_image(NBs, IoUs, FPs, FNs, percentiles, idx, config["plot_parameters"])
    
    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        max_IoU_index = IoUs.index(max(IoUs))
        aggregated_results['images_max_IoUs'].append(IoUs[max_IoU_index])
        aggregated_results['images_FPs_at_max_IoU'].append(FPs[max_IoU_index])
        aggregated_results['images_FNs_at_max_IoU'].append(FNs[max_IoU_index])
        aggregated_results['images_nb_seeds'].append(nb_seeds)

    return aggregated_results

def run_model_dataset(learner, images_links, masks_links, config):
    aggregated_results = {}
    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        aggregated_results = {'images_max_IoUs': [], 'images_FPs_at_max_IoU': [], 'images_FNs_at_max_IoU': [],
                              'images_percentiles': [], 'images_nb_seeds': []}
    total_number_images = int(config["training_parameters"]["TRAIN_RATIO"] * len(images_links))
    for idx in range(total_number_images):
        print("-------------------------------------", idx + 1, "/", total_number_images)
        if run_model_single_image(learner, images_links[idx], masks_links[idx], idx, aggregated_results, config) is None:
            continue
    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        plotandSaveAggregatedResults(config["plot_parameters"]["FOLDER_FOR_FINAL_RESULTS"], aggregated_results)


if __name__ == "__main__":
    print("Print")
    config = load_config("./config/base_config.json", "./config/setting1_config.json")
    config["model_parameters"]["DEVICE"] = check_environment()

    print("Training")
    print(config["learner_parameters"])
    sam = load_sam(config["model_parameters"])  
    learner = setup_learner(config["learner_parameters"], sam) 
    # images_links, masks_links = retrieveLinksForOneImageInEachFolder(config["data_parameters"])   
    images_links, masks_links = retrieveLinksForFolder(config["data_parameters"])
    run_model_dataset(learner, images_links, masks_links, config)