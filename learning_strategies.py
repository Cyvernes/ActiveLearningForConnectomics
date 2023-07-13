import numpy as np
import warnings


def basicLS(learner) -> None:
    learner.predictor.use_dropout = False
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords=np.array(learner.input_points),
        point_labels=np.array(learner.input_labels),
        mask_input=learner.logits if learner.use_previous_logits else None,
        multimask_output=False,
        return_logits=True,
    )
    learner.evidence = evidence.squeeze()  # image format
    learner.cp_mask = learner.evidence > learner.predictor.model.mask_threshold


def oneSeedForOneMaskLS(learner) -> None:
    learner.predictor.use_dropout = False
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
    if learner.nb_seed_used <= 1:
        warnings.warn(
            "should correct learner.masks_from_single_seeds when a false label is in a mask from a single seed"
        )
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords=np.array(learner.input_points),
        point_labels=np.array(learner.input_labels),
        mask_input=learner.logits if learner.use_previous_logits else None,
        multimask_output=False,
        return_logits=True,
    )
    learner.evidence = evidence.squeeze()  # image format
    learner.cp_mask = np.bitwise_or(
        learner.evidence > learner.predictor.model.mask_threshold,
        learner.masks_from_single_seeds,
    )


def oneSeedForOneMaskLSWithDropOut(learner) -> None:
    learner.predictor.use_dropout = True
    learner.predictor.setDropOutParameter(0.3)
    nb_tests = 20

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
    if len(learner.input_points) <= 1:
        warnings.warn(
            "should correct learner.masks_from_single_seeds when a false label is in a mask from a single seed"
        )
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
        learner.masks_from_single_seeds,
    )
    learner.predictor.use_dropout = False
