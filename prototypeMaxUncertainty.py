import sys
import cv2
import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np

SAVE_INTERMEDIATE_RESULTS = True
SAVE_FIRST_SEED = True

def centroid(mask : np.array):
    a, b = np.where(mask)
    return(int(np.mean(a)), int(np.mean(b)))

def swap(t : list):
    return([t[1], t[0]])

def IoU(mask1, mask2):
    intersection = np.sum(np.bitwise_and(mask1, mask2))
    union = np.sum(np.bitwise_or(mask1, mask2))
    return(intersection / union)

def getLabel(seed, GT_mask): #seed should be given in the SAM predict format and not in the image format
    return(GT_mask[seed[1], seed[0]])

def getEvidence(seed, evidence):
    return(evidence[0,seed[0], seed[1]])

def sigmoid(x):
    return(1 /(1 + np.exp(-x)))

def findVisualCenter(mask):
    dist = cv2.distanceTransform(mask.astype("uint8"), cv2.DIST_L2, 3)
    return(np.unravel_index(np.argmax(dist), dist.shape))

def findNewSeedWithMaxEvidence(evidence):
    _ , cx, cy = np.unravel_index(np.argmax(evidence), evidence.shape)
    return([cx, cy])

def uncertaintyKL(x):
    abs_x = np.abs(x)
    return(np.log( 1 + np.exp(-abs_x)))

def findNewSeedWithMaxUncertainty(evidence):
    return(np.unravel_index(np.argmax(uncertaintyKL(evidence)), evidence.shape))
    
if __name__ == "__main__":
    print('-------------------------------')
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    print('-------------------------------')
    print("Loading SAM...")
    
    sam_checkpoint = "SAM_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print('-------------------------------')
    print("Loading image...")
    image = cv2.imread("data/image.tiff")
    GT_mask = np.any(cv2.imread("data/GT_mask.tiff") != [0, 0, 0] , axis = -1)
    
    imagewGT = image.copy()
    imagewGT[GT_mask] = 0.7*image[GT_mask] + 0.3*np.array([75, 0, 125])
    plt.imshow(imagewGT)
    plt.savefig("Image with GT.png")
    plt.clf()
    
    
    print('-------------------------------')
    print("Learning ...")
    
    #find first seed
    se_masks = sorted(mask_generator.generate(image), key = lambda mask: mask['predicted_iou'])
    nb_seeds = len(se_masks)
    se_mask = se_masks.pop(-1)['segmentation'] #pop(-2) to have a mitochondrion ans pop(-1) to have the background
    first_seed = swap(findVisualCenter(se_mask))
    
    potential_seeds = [swap(centroid(mask['segmentation'])) for mask in se_masks]
    
    if SAVE_FIRST_SEED:
        imagewsem = image.copy()
        imagewsem[se_mask] = 0.7*image[se_mask] + 0.3*np.array([75, 0, 125])
        plt.imshow(imagewsem)
        plt.title('First seed and first mask')
        plt.scatter([first_seed[0]], [first_seed[1]])
        plt.savefig("Image with first seed.png")
        plt.clf()
    
    

    cp_mask = se_mask.copy()#current predicted mask
    IoUs = [IoU(GT_mask, cp_mask)]
    
    
    input_points = []
    input_labels = []
    new_seed = first_seed.copy()
    
    
    look_for_first_GT_mitochondria = True
    
    i = 0
    predictor.set_image(image)
    while i < nb_seeds != 0:
        input_points.append(new_seed)
        input_labels.append(getLabel(new_seed, GT_mask))
        
        if getLabel(new_seed, GT_mask):
            look_for_first_GT_mitochondria = False
        
        evidence, score, logit = predictor.predict(
            point_coords = np.array(input_points),
            point_labels = np.array(input_labels),
            multimask_output = False,
            return_logits = True,
        )
        cp_maskk = evidence > predictor.model.mask_threshold
        cp_mask = np.any(cp_maskk.transpose(1, 2, 0) != [False], axis = -1)

        #find new seed
        if look_for_first_GT_mitochondria:
            potential_seeds = sorted(potential_seeds, key = lambda seed : getEvidence(seed, evidence))
            new_seed = potential_seeds.pop(-1)
        else:
            new_seed = swap(findNewSeedWithMaxUncertainty(evidence))
            
        #draw i-th prediction
        if SAVE_INTERMEDIATE_RESULTS:
            fig, axs = plt.subplots(2,2,  layout='constrained')
            
            prediction_i = image.copy()
            prediction_i[cp_mask] = 0.7*image[cp_mask] + 0.3*np.array([75, 0, 125])
            axs[0, 0].imshow(prediction_i)
            axs[0, 0].scatter([new_seed[0]], [new_seed[1]], color = "green" if getLabel(new_seed, GT_mask) else "red")
            axs[0, 0].title.set_text("Image with mask and new seed")

            im1 = axs[0, 1].imshow(evidence.transpose(1, 2, 0))
            axs[0, 1].title.set_text("Evidence")

            uncertainty = uncertaintyKL(evidence.transpose(1, 2, 0))
            im2 = axs[1, 0].imshow(uncertainty)
            axs[1, 0].title.set_text("Uncertainty")
            
            axs[1, 1].plot(IoUs)
                  
            fig.colorbar(im1, ax = axs[0, 1], location = "right",  shrink = 1)
            fig.colorbar(im2, ax = axs[1, 0], location = "right",  shrink = 1)
            
            fig.savefig(f"Results n° {i}.png")
            plt.close(fig)

        
            
        
        """
        imagewcnt = image.copy()
        cv2.drawContours(imagewcnt, [max_contour], 0, (0,255,0), 3)
        plt.imshow(imagewcnt)
        plt.scatter([cx], [cy], color = "red")
        plt.savefig("image with first contour.png")
        plt.clf()
        """
        IoUs.append(IoU(cp_mask, GT_mask))
        i += 1

    
    plt.plot(list(range(1,len(IoUs) + 1 )), IoUs)
    plt.xlabel("Nb of seeds")
    plt.ylabel('IoU')
    plt.savefig('IoU.png')
        
        
    
    
    
    
    
    
