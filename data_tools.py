""" This module provides functionality for retrieving links to images and masks stored in specified directories. 
"""
import os
import glob
import random
from typing import Dict, Tuple, List, Any

random.seed(42)

def retrieveLinksForFolder(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """This function retrieves the links for all images and their corresponding masks 
    in a specified folder. 

    :param config: Configuration dictionary with keys "IMAGE_FOLDER" and "NUMBER_OF_IMAGES"
    :type config: Dict[str, Any]
    :return: Lists of image and mask filepaths
    :rtype: Tuple[List[str], List[str]]
    """
    images_links = []
    masks_links = []

    img_dir = os.path.join(config["IMAGE_FOLDER"], "images")
    mask_dir = os.path.join(config["IMAGE_FOLDER"], "masks")

    img_files = glob.glob(os.path.join(img_dir, "*.tiff"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.tiff"))

    if img_files and mask_files:
        img_files.sort()
        mask_files.sort()
        # images_links.extend(img_files)  # Add all image files to the list
        # masks_links.extend(mask_files)  # Add all mask files to the list

        combined = list(zip(img_files, mask_files))
        combined_sample = random.sample(combined, config["NUMBER_OF_IMAGES"]) # Randomly sample images
        images_links, masks_links = zip(*combined_sample)

    return images_links, masks_links


def retrieveLinksForOneImageInEachFolder(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """This function retrieves the links for one image and its corresponding mask
    from each subfolder within a specified directory.

    :param config: Configuration dictionary with key "BASE_DIR"
    :type config: Dict[str, Any]
    :return: Lists of image and mask filepaths
    :rtype: Tuple[List[str], List[str]]
    """
    subdirs = [
        os.path.join(config["BASE_DIR"], d)
        for d in os.listdir(config["BASE_DIR"])
        if os.path.isdir(os.path.join(config["BASE_DIR"], d))
    ]

    images_links = []
    masks_links = []

    for subdir in subdirs:
        img_dir = os.path.join(subdir, "images")
        mask_dir = os.path.join(subdir, "masks")

        img_files = glob.glob(os.path.join(img_dir, "*.tiff"))
        mask_files = glob.glob(os.path.join(mask_dir, "*.tiff"))

        if img_files and mask_files:
            img_files.sort()
            mask_files.sort()
            images_links.append(img_files[0])
            masks_links.append(mask_files[0])

    return images_links, masks_links
