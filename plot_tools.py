
import matplotlib.pyplot as plt
import os
from tools import *

def plotAndSaveIntermediateResults(learner, new_seed, image, GT_mask, FOLDER_FOR_INTERMEDIATE_RESULTS, IoUs, FNs, FPs, i : int, idx : int, nb_seeds):
    title_fontsize = 25
    xtick_fontsize = 20
    ytick_fontsize = 20
    plots_linewidth = 3
    new_seed_s = 50
    cp_mask = learner.cp_mask
    fig, axes = plt.subplots(2, 3, layout = 'constrained')
    fig.set_figheight(15)
    fig.set_figwidth(23)
                
    prediction_i = image.copy()
    prediction_i[cp_mask] = 0.7*image[cp_mask] + 0.3*np.array([75, 0, 125])
    axes[0, 0].imshow(prediction_i)
    if i != nb_seeds - 1:
        axes[0, 0].scatter([new_seed[0]], [new_seed[1]], color = "green" if getLabel(new_seed, GT_mask) else "red", s = new_seed_s)
        axes[0, 0].set_title("Image with mask and new seed", fontsize = title_fontsize)
    else:
        axes[0, 0].set_title("Image with final mask", fontsize = title_fontsize)
    axes[0, 0].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[0, 0].tick_params(axis = "y", labelsize = ytick_fontsize) 

    im1 = axes[0, 1].imshow(learner.evidence)
    axes[0, 1].set_title("Evidence", fontsize = title_fontsize)
    axes[0, 1].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[0, 1].tick_params(axis = "y", labelsize = ytick_fontsize) 

    uncertainty = uncertaintyH(learner.evidence)
    im2 = axes[0, 2].imshow(uncertainty)
    axes[0, 2].set_title("Uncertainty", fontsize = title_fontsize)
    axes[0, 2].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[0, 2].tick_params(axis = "y", labelsize = ytick_fontsize) 
                
    axes[1, 0].plot(IoUs, linewidth = plots_linewidth)
    axes[1, 0].set_title("Intersection over Union (IoU)",  fontsize = title_fontsize)
    axes[1, 0].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[1, 0].tick_params(axis = "y", labelsize = ytick_fontsize) 
                
    axes[1, 1].plot(FPs, linewidth = plots_linewidth)
    axes[1, 1].set_title("False Positives (FP)", fontsize = title_fontsize)
    axes[1, 1].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[1, 1].tick_params(axis = "y", labelsize = ytick_fontsize) 
                
    axes[1, 2].plot(FNs, linewidth = plots_linewidth)
    axes[1, 2].set_title("False Negatives (FN)", fontsize = title_fontsize)
    axes[1, 2].tick_params(axis = "x", labelsize = xtick_fontsize) 
    axes[1, 2].tick_params(axis = "y", labelsize = ytick_fontsize) 
                
    fig.colorbar(im1, ax = axes[0, 1], location = "right",  shrink = 1)
    fig.colorbar(im2, ax = axes[0, 2], location = "right",  shrink = 1)
                
    fig.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Results n° {i} of image n°{idx}.png"))
    plt.close(fig)


def plotAndSaveImageWithFirstSeed(image, first_mask, first_seed, FOLDER_FOR_INTERMEDIATE_RESULTS, idx : int):
    imagewsem = image.copy()
    imagewsem[first_mask] = 0.7*image[first_mask] + 0.3*np.array([75, 0, 125])
    plt.imshow(imagewsem)
    plt.title('First seed from this mask (from segment everything)')
    plt.scatter([first_seed[0]], [first_seed[1]])
    plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Image n°{idx} with first seed.png"))
    plt.clf()

def plotAndSaveImageWithGT(image, GT_mask, FOLDER_FOR_INTERMEDIATE_RESULTS, idx : int):
    imagewGT = image.copy()
    imagewGT[GT_mask] = 0.7*image[GT_mask] + 0.3*np.array([75, 0, 125])
    plt.imshow(imagewGT)
    plt.title(f"Image n°{idx} with GT")
    plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f"Image n°{idx} with GT.png"))
    plt.clf()
    
def plotAndSaveFinalIoUEvolution(IoUs, FOLDER_FOR_INTERMEDIATE_RESULTS, idx : int):
    plt.plot(list(range(1,len(IoUs) + 1 )), IoUs)
    plt.xlabel("Nb of seeds")
    plt.ylabel('IoU')
    plt.savefig(os.path.join(FOLDER_FOR_INTERMEDIATE_RESULTS, f'IoU_{idx}.png'))
    plt.clf()