import os
import glob


def retrieveLinksForOneImageInEachFolder():  # R
    # Retrieve one image from each folder
    base_dir = "../../data/cem_mitolab"

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
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
