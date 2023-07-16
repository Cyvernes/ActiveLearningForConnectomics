import numpy as np
import warnings
from tools import *
from plot_tools import *


def basicLS(learner) -> None:
    """
    Setting the global parameters  of the learning strategy
    """
    learner.predictor.use_dropout = False
    """
    Segmentaing and Estimating Uncertainties all at once
    """
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords=np.array(learner.input_points),
        point_labels=np.array(learner.input_labels),
        mask_input=learner.logits if learner.use_previous_logits else None,
        multimask_output=False,
        return_logits=True,
    )
    learner.evidence = evidence.squeeze()  # image format

    """
    Final prediction
    """
    learner.cp_mask = learner.evidence > learner.predictor.model.mask_threshold


def oneSeedForOneMaskLS(learner) -> None:
    """
    Setting the global parameters  of the learning strategy
    """
    learner.predictor.use_dropout = False

    """
    Segmentation strategy
    """
    for idx in range(learner.nb_seed_used, len(learner.input_points)):
        if learner.input_labels[idx]:
            evidence, scores, logits = learner.predictor.predict(
                point_coords=np.array([learner.input_points[idx]]),
                point_labels=np.array([learner.input_labels[idx]]),
                mask_input=None,
                multimask_output=False,
                return_logits=True,
            )
            evidence = evidence.squeeze()  # image format
            mask_from_seed = evidence > learner.predictor.model.mask_threshold
            learner.masks_from_single_seeds[mask_from_seed] = 1

    """
    Uncertainty strategy
    """
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords=np.array(learner.input_points),
        point_labels=np.array(learner.input_labels),
        mask_input=learner.logits if learner.use_previous_logits else None,
        multimask_output=False,
        return_logits=True,
    )
    learner.evidence = evidence.squeeze()  # image format

    """
    Final prediction
    """
    learner.cp_mask = np.bitwise_or(
        learner.evidence > learner.predictor.model.mask_threshold,
        learner.masks_from_single_seeds >= 0,
    )


def oneSeedForOneMaskLSWithDropOut(learner) -> None:
    """
    Setting the global parameters  of the learning strategy
    """
    learner.predictor.use_dropout = True
    learner.predictor.setDropOutParameter(0.3)
    nb_tests = 20

    if len(learner.input_points) - learner.nb_seed_used > 1:
        Warning.warn("This learning strategy has not been implemented to use multiple new seeds")

    """
    Segmentation strategy
    """
    if learner.input_labels[-1]:
        evidence, scores, logits = learner.predictor.predict(
            point_coords=np.array([learner.input_points[-1]]),
            point_labels=np.array([learner.input_labels[-1]]),
            mask_input=None,
            multimask_output=False,
            return_logits=True,
        )
        evidence = evidence.squeeze()  # image format
        mask_from_seed = evidence > learner.predictor.model.mask_threshold
        learner.masks_from_single_seeds[mask_from_seed] = 1

    """
    Uncertainty strategy
    """
    evidence = np.zeros_like(learner.evidence)
    if learner.use_previous_logits:
        logits = np.zeros_like(learner.logits)
    for idx in range(nb_tests):
        current_evidence, current_scores, current_logits = learner.predictor.predict(
            point_coords=np.array(learner.input_points),
            point_labels=np.array(learner.input_labels),
            mask_input=learner.logits if learner.use_previous_logits else None,
            multimask_output=False,
            return_logits=True,
        )
        evidence += current_evidence.squeeze()  # image format
        if learner.use_previous_logits:
            logits += current_logits
    learner.evidence = evidence / nb_tests
    if learner.use_previous_logits:
        learner.logits = logits / nb_tests
    learner.cp_mask = np.bitwise_or(
        learner.evidence > learner.predictor.model.mask_threshold,
        learner.masks_from_single_seeds >= 0,
    )

    """
    Final prediction
    """
    learner.predictor.use_dropout = False  # avoid bugs with change of strategies


def FewSeedsForOneMaskLS(learner) -> None:
    """
    Setting the global parameters  of the learning strategy
    """
    learner.predictor.use_dropout = False

    """
    Segmentation strategy
    """
    for idx in range(learner.nb_seed_used, len(learner.input_points)):
        # Computing the object around this seed
        seed = learner.input_points[idx]
        evidence_temp, _, _ = learner.predictor.predict(
            point_coords=np.array([seed]),
            point_labels=np.array([True]),
            mask_input=None,
            multimask_output=False,
            return_logits=True,
        )
        evidence_temp = evidence_temp.squeeze()  # image format
        mask_temp = evidence_temp > learner.predictor.model.mask_threshold
        previous_prediction = learner.masks_from_single_seeds[seed[1], seed[0]] 
        if previous_prediction > 0:
            #visual_center = findVisualCenter(intersection_of_segmentation)
            #value_at_visual_center = learner.masks_from_single_seeds[visual_center[0], visual_center[1]]
            corresponding_subset = abs(previous_prediction) - 1
            learner.seed_subsets[corresponding_subset].append(learner.input_points[idx])
            learner.label_subsets[corresponding_subset].append(learner.input_labels[idx])
            new_segmentation, _, _ = learner.predictor.predict(
                point_coords=np.array(learner.seed_subsets[corresponding_subset]),
                point_labels=np.array(learner.label_subsets[corresponding_subset]),
                mask_input=None,
                multimask_output=False,
                return_logits=False,
            )
            new_segmentation = new_segmentation.squeeze()
            blob_to_change = learner.masks_from_single_seeds == previous_prediction
            learner.masks_from_single_seeds[blob_to_change] = new_segmentation[blob_to_change]

        else:
            learner.masks_from_single_seeds[np.logical_and(mask_temp, learner.masks_from_single_seeds <= 0)] = (
                (len(learner.seed_subsets) + 1) if learner.input_labels[idx] else -(len(learner.seed_subsets) + 1)
            )
            learner.seed_subsets.append([learner.input_points[idx]])
            learner.label_subsets.append([learner.input_labels[idx]])
        
        if learner.count == 80:
            raise RuntimeError("stop")

        

    """
    Uncertainty strategy
    """
    evidence, _, learner.logits = learner.predictor.predict(
        point_coords=np.array(learner.input_points),
        point_labels=np.array(learner.input_labels),
        mask_input=learner.logits if learner.use_previous_logits else None,
        multimask_output=False,
        return_logits=True,
    )
    learner.evidence = evidence.squeeze()  # image format

    """
    Final prediction
    """
    #learner.cp_mask = np.bitwise_or(learner.evidence > learner.predictor.model.mask_threshold,learner.masks_from_single_seeds > 0,)
    learner.cp_mask = learner.masks_from_single_seeds >0
