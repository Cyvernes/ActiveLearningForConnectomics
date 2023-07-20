import os
import glob


def retrieveLinksForFolder(config):
    images_links = []
    masks_links = []

    img_dir = os.path.join(config["IMAGE_FOLDER"], "images")
    mask_dir = os.path.join(config["IMAGE_FOLDER"], "masks")

    img_files = glob.glob(os.path.join(img_dir, "*.tiff"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.tiff"))

    if img_files and mask_files:
        img_files.sort()
        mask_files.sort()
        images_links.extend(img_files)  # Add all image files to the list
        masks_links.extend(mask_files)  # Add all mask files to the list

    return images_links, masks_links


def retrieveLinksForOneImageInEachFolder(config):  # R
    # Retrieve one image from each folder
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
