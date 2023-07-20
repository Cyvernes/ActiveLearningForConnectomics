import numpy as np
import matplotlib.pyplot as plt
import os
from tools import *
from Learners import ActiveLearningSAM
import scipy.ndimage


def savePercentiles(
    learner: ActiveLearningSAM, percentiles_points: list, percentiles: list
) -> None:
    """This auxiliary function computes and saves percentiles at points from percentiles_points into percentiles

    Args:
        learner (ActiveLearningSAM): Learner
        percentiles_points (list): list containing all percentiles to be calculated
        percentiles (list): List in which results are concatenated
    """
    uncertainty = learner.uncertainty_function(learner.evidence)
    for i, p in enumerate(percentiles_points):
        percentiles[i].append(np.percentile(uncertainty, p))


def plotAndSaveIntermediateResults(
    folder: str,
    learner: ActiveLearningSAM,
    next_seeds: list,
    image: np.ndarray,
    GT_mask: np.ndarray,
    IoUs: list,
    FNs: list,
    FPs: list,
    i: int,
    idx: int,
    NBs: list,
) -> None:
    """This function is the main function for plotting intermediate results.
    The upper left picture represents the image, seeds and the current mask segmentation.
    Red dots represent background seeds, green dots represent foreground seeds and stars represent next seeds to be annotated with the corresponding color.

    Args:
        folder (str): Folder in which the figure will be saved
        learner (ActiveLearningSAM): Learner
        next_seeds (list): Next seeds to be annotated
        image (np.ndarray): Image
        GT_mask (np.ndarray): Ground truth mask
        IoUs (list): List of all IoUs (intersection over union) at each iteration
        FNs (list): List of all false negatives at each iteration
        FPs (list): List of all false positives at each iteration
        i (int): Index of the iteration
        idx (int): Index of the image in the dataset
        NBs (list): Number of seeds at each iteration until this one
    """
    title_fontsize = 25
    xtick_fontsize = 20
    ytick_fontsize = 20
    plots_linewidth = 3
    new_seed_s = 180
    seed_s = 130
    mksize = 17
    cp_mask = learner.cp_mask.astype("bool")
    # cp_mask = learner.masks_from_single_seeds.astype("bool")
    fig, axes = plt.subplots(2, 3, layout="constrained")
    fig.set_figheight(15)
    fig.set_figwidth(23)

    prediction_i = image.copy()
    prediction_i[cp_mask] = 0.7 * image[cp_mask] + 0.3 * np.array([75, 0, 125])
    axes[0, 0].imshow(prediction_i)
    if next_seeds != []:
        green_seeds = [
            s
            for s in learner.input_points
            if (getLabel(s, GT_mask) and not (s in next_seeds))
        ]
        red_seeds = [
            s
            for s in learner.input_points
            if (not (getLabel(s, GT_mask)) and not (s in next_seeds))
        ]
        green_new_seeds = [ns for ns in next_seeds if getLabel(ns, GT_mask)]
        red_new_seeds = [ns for ns in next_seeds if (not getLabel(ns, GT_mask))]
        axes[0, 0].scatter(
            [ns[0] for ns in green_new_seeds],
            [ns[1] for ns in green_new_seeds],
            color="green",
            marker=(5, 2),
            s=new_seed_s,
        )
        axes[0, 0].scatter(
            [ns[0] for ns in red_new_seeds],
            [ns[1] for ns in red_new_seeds],
            color="red",
            marker=(5, 2),
            s=new_seed_s,
        )
        axes[0, 0].scatter(
            [s[0] for s in green_seeds],
            [s[1] for s in green_seeds],
            color="green",
            s=seed_s,
        )
        axes[0, 0].scatter(
            [s[0] for s in red_seeds], [s[1] for s in red_seeds], color="red", s=seed_s
        )
        axes[0, 0].set_title("Current segmentation and seeds", fontsize=title_fontsize)
    else:  # last iteration
        green_seeds = [s for s in learner.input_points if (getLabel(s, GT_mask))]
        red_seeds = [s for s in learner.input_points if (not (getLabel(s, GT_mask)))]
        axes[0, 0].scatter(
            [s[0] for s in green_seeds],
            [s[1] for s in green_seeds],
            color="green",
            s=seed_s,
        )
        axes[0, 0].scatter(
            [s[0] for s in red_seeds], [s[1] for s in red_seeds], color="red", s=seed_s
        )
        axes[0, 0].set_title("Final segmentation and seeds", fontsize=title_fontsize)
    axes[0, 0].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[0, 0].tick_params(axis="y", labelsize=ytick_fontsize)

    im1 = axes[0, 1].imshow(learner.evidence)
    axes[0, 1].set_title("Evidence", fontsize=title_fontsize)
    axes[0, 1].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[0, 1].tick_params(axis="y", labelsize=ytick_fontsize)

    uncertainty = learner.uncertainty_function(learner.evidence)
    uncertainty = learner.filtering_function(learner, uncertainty)
    im2 = axes[0, 2].imshow(uncertainty)
    axes[0, 2].set_title("Uncertainty (filtered)", fontsize=title_fontsize)
    axes[0, 2].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[0, 2].tick_params(axis="y", labelsize=ytick_fontsize)

    green_xs = [
        nn
        for i, nn in enumerate(NBs)
        if (True in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    red_xs = [
        nn
        for i, nn in enumerate(NBs)
        if (False in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    green_IoUs_ys = [
        IoUs[i]
        for i in range(len(NBs))
        if (True in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    red_IoUs_ys = [
        IoUs[i]
        for i in range(len(NBs))
        if (False in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]

    axes[1, 0].plot(NBs, IoUs, linewidth=plots_linewidth)
    axes[1, 0].set_title("Intersection over Union (IoU)", fontsize=title_fontsize)
    axes[1, 0].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[1, 0].tick_params(axis="y", labelsize=ytick_fontsize)
    axes[1, 0].plot(
        green_xs,
        green_IoUs_ys,
        color="green",
        marker="o",
        lw=0,
        fillstyle="full",
        markersize=mksize,
    )
    axes[1, 0].plot(
        red_xs,
        red_IoUs_ys,
        color="red",
        marker="o",
        lw=0,
        fillstyle="none",
        markersize=mksize,
    )
    for x in learner.idx_when_strat_has_changed:
        axes[1, 0].axvline(x=x, color="gray")

    green_FPs_ys = [
        FPs[i]
        for i in range(len(NBs))
        if (True in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    red_FPs_ys = [
        FPs[i]
        for i in range(len(NBs))
        if (False in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    axes[1, 1].plot(NBs, FPs, linewidth=plots_linewidth)
    axes[1, 1].set_title("False Positives (FP)", fontsize=title_fontsize)
    axes[1, 1].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[1, 1].tick_params(axis="y", labelsize=ytick_fontsize)
    axes[1, 1].plot(
        green_xs,
        green_FPs_ys,
        color="green",
        marker="o",
        lw=0,
        fillstyle="full",
        markersize=mksize,
    )
    axes[1, 1].plot(
        red_xs,
        red_FPs_ys,
        color="red",
        marker="o",
        lw=0,
        fillstyle="none",
        markersize=mksize,
    )
    for x in learner.idx_when_strat_has_changed:
        axes[1, 1].axvline(x=x, color="gray")

    green_FNs_ys = [
        FNs[i]
        for i in range(len(NBs))
        if (True in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    red_FNs_ys = [
        FNs[i]
        for i in range(len(NBs))
        if (False in learner.input_labels[0 if i == 0 else NBs[i - 1] : NBs[i]])
    ]
    axes[1, 2].plot(NBs, FNs, linewidth=plots_linewidth)
    axes[1, 2].set_title("False Negatives (FN)", fontsize=title_fontsize)
    axes[1, 2].tick_params(axis="x", labelsize=xtick_fontsize)
    axes[1, 2].tick_params(axis="y", labelsize=ytick_fontsize)
    axes[1, 2].plot(
        green_xs,
        green_FNs_ys,
        color="green",
        marker="o",
        lw=0,
        fillstyle="full",
        markersize=mksize,
    )
    axes[1, 2].plot(
        red_xs,
        red_FNs_ys,
        color="red",
        marker="o",
        lw=0,
        fillstyle="none",
        markersize=mksize,
    )
    for x in learner.idx_when_strat_has_changed:
        axes[1, 2].axvline(x=x, color="gray")

    fig.colorbar(im1, ax=axes[0, 1], location="right", shrink=1)
    fig.colorbar(im2, ax=axes[0, 2], location="right", shrink=1)

    fig.savefig(os.path.join(folder, f"Results n° {i} of image n°{idx}.png"))
    plt.close(fig)


def plotAndSaveImageWithFirstSeed(
    folder: str, image: np.ndarray, first_seeds: list, idx: int
) -> None:
    """This function plots and saves the image and the first seeds used in the first iteration

    Args:
        folder (str): Folder in which the figure will be saved
        image (np.ndarray): Image
        first_seeds (list): First seeds
        idx (int): Index of the image in the dataset
    """
    imagewsem = image.copy()
    plt.imshow(imagewsem)
    plt.title("First seed from this mask (from segment everything)")
    plt.scatter([fs[0] for fs in first_seeds], [fs[1] for fs in first_seeds])
    plt.savefig(os.path.join(folder, f"Image n°{idx} with first seed.png"))
    plt.clf()


def plotAndSaveImageWithFirstSeedandFirstMask(
    folder: str, image: np.ndarray, first_mask: np.ndarray, first_seeds: list, idx: int
) -> None:
    """This function plots and save the image with the first seeds and the first mask

    Args:
        folder (str): Folder in which the figure will be saved
        image (np.ndarray): Image
        first_mask (np.ndarray): First mask
        first_seeds (list): First seeds
        idx (int): Index of the image in the dataset
    """
    imagewsem = image.copy()
    imagewsem[first_mask] = 0.7 * image[first_mask] + 0.3 * np.array([75, 0, 125])
    plt.imshow(imagewsem)
    plt.title("First seed from this mask (from segment everything)")
    plt.scatter([fs[0] for fs in first_seeds], [fs[1] for fs in first_seeds])
    plt.savefig(os.path.join(folder, f"Image n°{idx} with first seed.png"))
    plt.clf()


def plotAndSaveImageWithGT(
    folder: str, image: np.ndarray, GT_mask: np.ndarray, idx: int
) -> None:
    """This function plots and saves the image with the ground truth mask

    Args:
        folder (str): Folder in which the figure will be saved
        image (np.ndarray): Image
        GT_mask (np.ndarray): Ground truth mask
        idx (int): Index of the image in the dataset
    """
    imagewGT = image.copy()
    imagewGT[GT_mask] = 0.7 * image[GT_mask] + 0.3 * np.array([75, 0, 125])
    plt.imshow(imagewGT)
    plt.title(f"Image n°{idx} with GT")
    plt.savefig(os.path.join(folder, f"Image n°{idx} with GT.png"))
    plt.clf()


def plotAndSaveFinalIoUEvolution(folder: str, NBs: list, IoUs: list, idx: int):
    """This function plots the evolution of IoU during the experiment

    Args:
        folder (str): Folder in which the figure will be saved
        NBs (list): Number of seeds at each iteration
        IoUs (list): List of all IoUs (intersection over union) at each iteration
        idx (int): Index of the image in the dataset
    """
    plt.plot(NBs, IoUs)
    plt.xlabel("Nb of seeds")
    plt.ylabel("IoU")
    plt.savefig(os.path.join(folder, f"IoU_{idx}.png"))
    plt.clf()


def savePercentilesPlot(folder: str, NBs: list, percentiles: list, idx: int):
    """This function saves the evolution of percentiles during the experiment

    Args:
        folder (str): Folder in which the figure will be saved
        NBs (list): Number of seeds at each iteration
        percentiles (list): Evolution of percentiles during the experiment
        idx (int): Index of the image in the dataset
    """
    for i, history in enumerate(percentiles):
        plt.plot(NBs, history)
    plt.savefig(os.path.join(folder, f"Uncertainty percentiles evolution n°{idx}.png"))
    plt.clf()


def plotAndSave(arr: np.array, name: str):
    """This function plot and save an image

    Args:
        arr (np.array): The image to be plotted
        name (str): name of the output file
    """
    plt.imshow(arr)
    plt.savefig(name)
    plt.clf()


def plotandSaveAggregatedResults(folder: str, aggregated_results: dict):
    """This function plots and save the aggreagation of results on the whole dataset

    Args:
        folder (str): Folder in which the figure will be saved
        aggregated_results (dict): Aggregated results
    """

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Extract data from aggregated_results
    IoUs = aggregated_results["images_max_IoUs"]
    FPs = aggregated_results["images_FPs_at_max_IoU"]
    FNs = aggregated_results["images_FNs_at_max_IoU"]
    final_nb_seeds = aggregated_results["images_nb_seeds"]

    # Plot IoUs
    axs[0, 0].hist(IoUs, bins=10, edgecolor="black")
    axs[0, 0].set_title("Distribution of IoUs")
    axs[0, 0].set_xlabel("IoU")
    axs[0, 0].set_ylabel("Frequency")

    # Plot FPs
    axs[0, 1].hist(FPs, bins=10, edgecolor="black")
    axs[0, 1].set_title("Distribution of FPs")
    axs[0, 1].set_xlabel("FP")
    axs[0, 1].set_ylabel("Frequency")

    # Plot FNs
    axs[1, 0].hist(FNs, bins=10, edgecolor="black")
    axs[1, 0].set_title("Distribution of FNs")
    axs[1, 0].set_xlabel("FN")
    axs[1, 0].set_ylabel("Frequency")

    # Plot nb_seeds
    axs[1, 1].hist(final_nb_seeds, bins=10, edgecolor="black")
    axs[1, 1].set_title("Distribution of NB Seeds")
    axs[1, 1].set_xlabel("NB Seeds")
    axs[1, 1].set_ylabel("Frequency")

    # Add spacing between plots
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "Final Results"))
    plt.clf()
