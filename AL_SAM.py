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

SUBSET_SIZE = 1
TRAIN_RATIO = 1
LOAD_DATA_ONCE_FOR_ALL = True
CHOOSE_DATA_AT_RANDOM = False
SAVE_INTERMEDIATE_RESULTS = True
SAVE_IMAGE_WITH_GT = True
SAVE_FIRST_SEED = True
LEARNER_TYPE = "Basic" # {"Basic", "FPFN", "MaxUncertainty", "DistTransform", "UncertaintyPath"}

FILE_WITH_ALL_LINKS = "/n/home12/cyvernes/working_directory/cem_mitolab_dataset_links.json"
FOLDER_FOR_INTERMEDIATE_RESULTS = "working_directory/temp/pred_evol/"
SPECIFIC_IMAGE_LINKS = [ '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-R/images/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/jrc_ctl-id8-4_openorganelle/images/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/images/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-H/images/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/52f72Bc125o7v94Ep850_2D/images/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff' ]
SPECIFIC_MASK_LINKS =  [ '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-R/masks/Wei2020_MitoEM-R-ROI-x0-500_y3072-3584_z1024-1536-LOC-0_254_0-512_0-512.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/jrc_ctl-id8-4_openorganelle/masks/jrc_ctl-id8-4_openorganelle-ROI-x229-458_y2466-2690_z3365-3589-LOC-2_70-75_0-224_0-224.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/271N4JXZ0Ux7d61W39t6_3D/masks/271N4JXZ0Ux7d61W39t6_3D-LOC-0_32-37_0-115_0-224.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/Wei2020_MitoEM-H/masks/Wei2020_MitoEM-H-ROI-x0-500_y512-1024_z3072-3584-LOC-0_076_0-512_0-512.tiff',
                         '/n/home12/cyvernes/CEM/CEM-MitoLab/data/cem_mitolab/52f72Bc125o7v94Ep850_2D/masks/52f72Bc125o7v94Ep850_2D_img00607-LOC-2d-1792-2016_448-672.tiff'      ]

SPECIFIC_IMAGE_LINKS = [SPECIFIC_IMAGE_LINKS[3]]
SPECIFIC_MASK_LINKS  = [SPECIFIC_MASK_LINKS[3]]

if __name__ == "__main__":
    """
    Check the environment and the cuda device
    """
    print('-------------------------------------')
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    Loading data
    """
    print('-------------------------------------')
    print("Retrieving the data...")
    if CHOOSE_DATA_AT_RANDOM:
        with open(FILE_WITH_ALL_LINKS) as infile:
            dataset_links = json.load(infile)
            print("There is" if len(dataset_links['images']) == 1 else "There are", len(dataset_links['images']), "images available in the dataset.")
        subset_idx = random.sample(range(len(dataset_links['images'])), SUBSET_SIZE)
        images_links = [dataset_links['images'][idx] for idx in subset_idx]
        masks_links  = [dataset_links['masks'][idx]  for idx in subset_idx]
        del subset_idx
    else:
        images_links = SPECIFIC_IMAGE_LINKS
        masks_links  =  SPECIFIC_MASK_LINKS
    
    print("There is" if len(images_links) == 1 else "There are", len(images_links),"selected images.")

    if LOAD_DATA_ONCE_FOR_ALL:
        print("Loading the data once for all...")
        train_images = [cv2.imread(images_links[idx]) for idx in range(int(TRAIN_RATIO * len(images_links)))]
        test_images  = [cv2.imread(images_links[idx]) for idx in range(int(TRAIN_RATIO * len(images_links)), len(images_links))]
        train_masks  = [np.any(cv2.imread(masks_links[idx]) != [0,0,0], axis = -1) for idx in range(int(TRAIN_RATIO * len(images_links)))]
        test_masks   = [np.any(cv2.imread(masks_links[idx]) != [0,0,0], axis = -1) for idx in range(int(TRAIN_RATIO * len(images_links)), len(images_links))]
    
    """
    Loading SAM
    """
    print('-------------------------------------')
    print("Loading the model...")
    
    #Loading model weights
    sam_checkpoint = "/n/home12/cyvernes/working_directory/SAM_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device = device)

    
    """
    Training
    """
    print('-------------------------------------')
    print("Training...")
    
    #instancing a learning algorithm
    if LEARNER_TYPE == "Basic":
        learner = ActiveLearningSAM(sam)
    elif LEARNER_TYPE == "FPFN":
        learner = FPFNLearner(sam)
    elif LEARNER_TYPE == "MaxUncertainty":
        learner = MaxUncertaintyLearner(sam)
    elif LEARNER_TYPE == "DistTransform":
        learner = DistTransformLearner(sam)
    elif LEARNER_TYPE == "UncertaintyPath":
        learner = UncertaintyPathLearner(sam)
    else:
        raise ValueError('Unknown learner type')
    
    for idx in range(int(TRAIN_RATIO*len(images_links))):
        print('-------------------------------------', idx + 1, '/', int(TRAIN_RATIO*len(images_links)))
        #load data
        if LOAD_DATA_ONCE_FOR_ALL:
            image = train_images[idx]
            GT_mask = train_masks[idx]
        else:
            image = cv2.imread(images_links[idx])
            GT_mask = np.any(cv2.imread(masks_links[idx]) != [0,0,0], axis = -1)

        
        if SAVE_IMAGE_WITH_GT:
            imagewGT = image.copy()
            imagewGT[GT_mask] = 0.7*image[GT_mask] + 0.3*np.array([75, 0, 125])
            plt.imshow(imagewGT)
            plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Image n°{idx} with GT.png"))
            plt.clf()

        #print("Ground truth mask shape:", np.shape(GT_mask))
        learner.setData(image)
        if learner.needGroundTruth:
            learner.setGroundTruthMask(GT_mask)
        
        #Find the first point to query
        first_seed, nb_seeds, first_mask = learner.findFirstSeed()
        if len(first_seed) == 0:#avoid empty list
            print("No first seed was given")
            continue
        

        if SAVE_FIRST_SEED:
            imagewsem = image.copy()
            imagewsem[first_mask] = 0.7*image[first_mask] + 0.3*np.array([75, 0, 125])
            plt.imshow(imagewsem)
            plt.title('First seed from this mask (from segment everything)')
            plt.scatter([first_seed[0]], [first_seed[1]])
            plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Image n°{idx} with first seed.png"))
            plt.clf()

        
        # Main loop
        input_points = []
        input_labels = []
        IoUs = []
        FPs = []
        FNs = []
        new_seed = first_seed.copy()

        for i in range(nb_seeds):
            input_points.append(new_seed)
            input_labels.append(getLabel(new_seed, GT_mask))
            
            if getLabel(new_seed, GT_mask):
                look_for_first_GT_mitochondria = False
            
            learner.learn(input_points, input_labels)

            new_seed = learner.findNewSeed()
               
            IoUs.append(IoU(learner.cp_mask, GT_mask)) 
            FPs.append(FP(learner.cp_mask, GT_mask)) 
            FNs.append(FN(learner.cp_mask, GT_mask)) 
            #draw i-th prediction
            if SAVE_INTERMEDIATE_RESULTS:
                cp_mask = learner.cp_mask
                fig, axs = plt.subplots(2, 3, layout = 'constrained')
                fig.set_figheight(15)
                fig.set_figwidth(23)
                
                prediction_i = image.copy()
                prediction_i[cp_mask] = 0.7*image[cp_mask] + 0.3*np.array([75, 0, 125])
                axs[0, 0].imshow(prediction_i)
                axs[0, 0].scatter([new_seed[0]], [new_seed[1]], color = "green" if getLabel(new_seed, GT_mask) else "red")
                axs[0, 0].set_title("Image with mask and new seed", fontsize = 25)

                im1 = axs[0, 1].imshow(learner.evidence)
                axs[0, 1].set_title("Evidence", fontsize = 25)

                uncertainty = uncertaintyH(learner.evidence)
                im2 = axs[0, 2].imshow(uncertainty)
                axs[0, 2].set_title("Uncertainty", fontsize = 25)
                
                axs[1, 0].plot(IoUs)
                axs[1, 0].set_title("IoU",  fontsize = 25)
                
                axs[1, 1].plot(FPs)
                axs[1, 1].set_title("FP", fontsize = 25)
                
                axs[1, 2].plot(FNs)
                axs[1, 2].set_title("FN", fontsize = 25)
                
                fig.colorbar(im1, ax = axs[0, 1], location = "right",  shrink = 1)
                fig.colorbar(im2, ax = axs[0, 2], location = "right",  shrink = 1)
                
                fig.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Results n° {i} of image n°{idx}.png"))
                plt.close(fig)


        plt.plot(list(range(1,len(IoUs) + 1 )), IoUs)
        plt.xlabel("Nb of seeds")
        plt.ylabel('IoU')
        plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f'IoU_{idx}.png'))
        plt.clf()
        
        """
        Testing
        """
        print('-------------------------------------')
        print("Testing...")
        


        