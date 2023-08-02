import json
from typing import Dict, List, Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from segment_anything import sam_model_registry

import filters
import first_seeds_selector
import learning_strategies
import next_seeds_strategies
import strategy_selectors
import tools
from Learners import *
from learning_strategies import *
from data_tools import *
from plot_tools import *
from tools import *


def load_config(base_json_file: str, setting_json_file: str) -> Dict[str, Any]:
    """This function loads the configuration from base and setting JSON files.
    The setting JSON file updates the base configuration.

    :param base_json_file: Path to the base configuration JSON file
    :type base_json_file: str
    :param setting_json_file: Path to the setting configuration JSON file
    :type setting_json_file: str
    :return: Configuration dictionary
    :rtype: Dict[str, Any]
    """
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


def check_environment() -> torch.device:
    """This function checks the environment versions

    :return: Device to be used for computations
    :rtype: torch.device
    """
    print("-------------------------------------")
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam(config: Dict[str, str]) -> Any:
    """This function loads the Segment Anything Model

    :param config: Configuration dictionary that includes 'MODEL_WEIGHTS_PATH' for model checkpoint, 
                   'MODEL_TYPE' for the type of the model, and 'DEVICE' to specify the device to which 
                   the model will be loaded.
    :type config: Dict[str, str]
    :return: Loaded model
    :rtype: Any
    """
    print("-------------------------------------")
    print("Loading the model...")
    sam_checkpoint = config['MODEL_WEIGHTS_PATH']
    model_type = config["MODEL_TYPE"]
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=config["DEVICE"])
    return sam


def setup_learner(config: Dict[str, Any], sam: Any) -> ActiveLearningSAM:
    """This function sets up the learner based on the given configuration and the model.

    :param config: Configuration dictionary 
    :type config: Dict[str, Any]
    :param sam: The loaded model (usually SAM)
    :type sam: Any
    :return: learner
    :rtype: ActiveLearningSAM
    """
    learner_types = {"Active Learning": ActiveLearningSAM, "Pseudo Active Learning": PseudoActiveLearningSAM,
                     "FPFN": FPFNLearner, "Random": RandomLearner}
    learning_strategy_types = {"BasicLS": BasicLS, "OneSeedForOneMaskLS": OneSeedForOneMaskLS,
                               "OneSeedForOneMaskLSWithDropOut": OneSeedForOneMaskLSWithDropOut,
                               "FewSeedsForOneMaskLS": FewSeedsForOneMaskLS}
    print()
    try:
        LearnerClass = learner_types[config["LEARNER_TYPE"]]
    except KeyError:
        raise ValueError("Unknown learner type")
    return LearnerClass(sam, getattr(strategy_selectors, config["STRATEGY_SELECTOR"]),
                        [learning_strategy_types[strategy] for strategy in config["LEARNING_STRATEGIES"]],
                        # [getattr(learning_strategies, strategy)() for strategy in config["LEARNING_STRATEGIES"]],
                        getattr(first_seeds_selector, config["FIRST_SEEDS_SELECTOR"]),
                        [getattr(next_seeds_strategies, strategy) for strategy in config["SEED_SELECTION_STRATEGIES"]],
                        getattr(tools, config["UNCERTAINTY_FUNCTION_TYPE"]),
                        getattr(filters, config["FILTERING_FUNCTION"]),
                        getattr(filters, config["FILTERING_AUX_FONCTION"]),
                        use_previous_logits=config["USE_PREVIOUS_LOGITS"])


def find_first_seeds(learner: ActiveLearningSAM, image: np.ndarray, GT_mask: np.ndarray, idx: int, config: Dict[str, Any]) -> Tuple[Optional[List[Any]], Optional[int]]:
    """This function finds the first seeds based on the given image

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param image: Input image.
    :type image: np.ndarray
    :param GT_mask: Ground truth mask.
    :type GT_mask: np.ndarray
    :param idx: Index of image
    :type idx: int
    :param config: Configuration dictionary
    :type config: Dict[str, Any]
    :return: The first seeds found by the learner and the number of seeds. If no seed is found, return None for both.
    :rtype: Tuple[Optional[List[Any]], Optional[int]]
    """
    learner.setData(image)

    if learner.need_ground_truth:
        learner.setGroundTruthMask(GT_mask)
    print(learner)
    first_seeds, nb_seeds = learner.findFirstSeeds()

    if len(first_seeds) == 0:
        print("No first seed was given")
        return None, None

    if config["SAVE_FIRST_SEED"]:
        plotAndSaveImageWithFirstSeed(config["FOLDER_FOR_INTERMEDIATE_RESULTS"], image, first_seeds, idx)

    if config["SAVE_IMAGE_WITH_GT"]:
        plotAndSaveImageWithGT(config["FOLDER_FOR_INTERMEDIATE_RESULTS"], image, GT_mask, idx)
    return first_seeds, nb_seeds


def run_learning_process(learner: ActiveLearningSAM, GT_mask: np.ndarray, first_seeds: List[Any], nb_seeds: int, image: np.ndarray, idx: int, config: Dict[str, Any]) -> Dict[str, list]:
    """This function runs the learning process on the image using the given learner, ground truth mask, first seeds, number of seeds, index, and configuration.

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param GT_mask: Ground truth mask
    :type GT_mask: np.ndarray
    :param first_seeds: The first seeds to be used in the learning process
    :type first_seeds: List[Any]
    :param nb_seeds: Number of seeds
    :type nb_seeds: int
    :param image: Input image
    :type image: np.ndarray
    :param idx: Index of image
    :type idx: int
    :param config: Configuration dictionary 
    :type config: Dict[str, Any]
    :return: A dictionary containing lists of numbers of input points ('NBs'), intersection over union scores ('IoUs'), false positives ('FPs'), false negatives ('FNs'), and percentiles ('percentiles').
    :rtype: Dict[str, list]
    """
    input_points = []
    input_labels = []
    NBs = []
    metrics = {
        "IoUs": IoU,
        "FPs": FP,
        "FNs": FN,
        "DLs": DiceLoss
    }
    result = {name: [] for name in metrics.keys()}
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
        for name, metric in metrics.items():
            result[name].append(metric(learner.cp_mask, GT_mask))
        if nb_annotations < budget:
            next_seeds = learner.findNextSeeds()
        else:
            next_seeds = []
        count += 1
        if config["plot_parameters"]["SAVE_INTERMEDIATE_RESULTS"]:
            plotAndSaveIntermediateResults(config["plot_parameters"]["FOLDER_FOR_INTERMEDIATE_RESULTS"], learner,
                                           next_seeds, image, GT_mask, result["IoUs"], result["FNs"], result["FPs"],
                                           count, idx, NBs)
        if nb_annotations == budget - 1 and config["plot_parameters"]["SAVE_FINAL_RESULT"]:
            plotAndSaveIntermediateResults(config["plot_parameters"]["FOLDER_FOR_INTERMEDIATE_RESULTS"], learner,
                                           next_seeds, image, GT_mask, result["IoUs"], result["FNs"], result["FPs"],
                                           count, idx, NBs)
        if config["plot_parameters"]["SAVE_UNCERTAINTY_PERCENTILES"]:
            savePercentiles(learner, percentiles_points, percentiles)
    
    result["percentiles"] = percentiles
    result["NBs"] = NBs
    return result



def plots_for_single_image(NBs: List, IoUs: List, percentiles: List, idx: int, config: Dict[str, Any]) -> None:
    """This function creates and saves plots for a single image.

    :param NBs: List of numbers of input points.
    :type NBs: List
    :param IoUs: List of Intersection over Union scores.
    :type IoUs: List
    :param percentiles: List of percentile values.
    :type percentiles: List
    :param idx: Index of image.
    :type idx: int
    :param config: Configuration dictionary.
    :type config: Dict[str, Any]
    """
    max_IoU = max(IoUs)
    for i, iou in enumerate(IoUs):
        if iou >= 0.9 * max_IoU:
            print(f"Used {i} iterations to reach 90% of max IoU")
            break
    if config["SAVE_UNCERTAINTY_PERCENTILES"]:
        savePercentilesPlot(config["FOLDER_FOR_FINAL_RESULTS"], NBs, percentiles, idx)

    if config["SAVE_FINAL_IOU_EVOLUTION"]:
        plotAndSaveFinalIoUEvolution(config["FOLDER_FOR_FINAL_RESULTS"], NBs, IoUs, idx)


def run_model_single_image(learner: ActiveLearningSAM, image_link: str, mask_link: str, idx: int, aggregated_results: Dict[str, list], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """This function runs the model on a single image

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param image_link: Path to the image file
    :type image_link: str
    :param mask_link: Path to the mask file
    :type mask_link: str
    :param idx: Index of the image
    :type idx: int
    :param aggregated_results: A dictionary containing results from previous runs
    :type aggregated_results: Dict[str, list]
    :param config: Configuration dictionary
    :type config: Dict[str, Any]
    :return: Updated aggregated_results if first seeds are found; None otherwise
    :rtype: Optional[Dict[str, Any]]
    """
    image = cv2.imread(image_link)
    GT_mask = np.any(cv2.imread(mask_link) != [0, 0, 0], axis=-1)

    first_seeds, nb_seeds = find_first_seeds(learner, image, GT_mask, idx, config["plot_parameters"])
    if first_seeds is None: return None
    result = run_learning_process(learner, GT_mask, first_seeds, nb_seeds, image, idx, config)
    plots_for_single_image(result['NBs'], result['IoUs'], result['percentiles'], idx, config["plot_parameters"])

    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        max_IoU_index = result['IoUs'].index(max(result['IoUs']))
        aggregated_results['images_max_IoUs'].append(result['IoUs'][max_IoU_index])
        aggregated_results['images_FPs_at_max_IoU'].append(result['FPs'][max_IoU_index])
        aggregated_results['images_FNs_at_max_IoU'].append(result['FNs'][max_IoU_index])
        aggregated_results['images_DLs_at_max_IoU'].append(result['DLs'][max_IoU_index])
        aggregated_results['images_nb_seeds'].append(nb_seeds)
        aggregated_results['images_max_IoUs_index'].append(max_IoU_index)
        print(max_IoU_index)

    return aggregated_results


def run_model_dataset(learner: ActiveLearningSAM, images_links: List[str], masks_links: List[str], config: Dict[str, Any]) -> None:
    """This function runs the model on a dataset

    :param learner: Learner
    :type learner: ActiveLearningSAM
    :param images_links: List of paths to the image files
    :type images_links: List[str]
    :param masks_links: List of paths to the mask files
    :type masks_links: List[str]
    :param config: Configuration dictionary
    :type config: Dict[str, Any]
    """
    aggregated_results = {}
    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        aggregated_results = {'images_max_IoUs': [], 'images_FPs_at_max_IoU': [], 'images_FNs_at_max_IoU': [],
                              'images_DLs_at_max_IoU': [], 'images_percentiles': [],  'images_nb_seeds': [], 'images_max_IoUs_index': []}
    total_number_images = int(config["training_parameters"]["TRAIN_RATIO"] * len(images_links))
    for idx in range(total_number_images):
        print("-------------------------------------", idx + 1, "/", total_number_images)
        if run_model_single_image(learner, images_links[idx], masks_links[idx], idx, aggregated_results,
                                  config) is None:
            continue
    print(aggregated_results)
    if config["plot_parameters"]["SAVE_AGGREGATED_RESULTS"]:
        plotandSaveAggregatedResults(config["plot_parameters"]["FOLDER_FOR_FINAL_RESULTS"], aggregated_results)


if __name__ == "__main__":
    """Main function of the pipeline to run on dataset
    """
    if len(sys.argv) > 1:
        override_config_path = sys.argv[1]
    else:
        override_config_path = "./config/base_config.json"
    
    print("Print")
    config = load_config("./config/base_config.json", override_config_path)
    config["model_parameters"]["DEVICE"] = check_environment()

    print("Training")
    sam = load_sam(config["model_parameters"])
    print(config["learner_parameters"])
    print(config["training_parameters"])
    learner = setup_learner(config["learner_parameters"], sam)

    images_links, masks_links = retrieveLinksForFolder(config["data_parameters"])
    run_model_dataset(learner, images_links, masks_links, config)