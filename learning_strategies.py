import numpy as np

def basicLS(learner) -> None:
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords = np.array(learner.input_points),
        point_labels = np.array(learner.input_labels),
        mask_input = learner.logits if learner.use_previous_logits else None,
        multimask_output = False,
        return_logits = True,
    )
    learner.evidence = evidence.squeeze() #image format
    learner.cp_mask = learner.evidence > learner.predictor.model.mask_threshold
    
def oneSeedForOneMaskLS(learner) -> None:
    if learner.input_labels[-1]:
        evidence, scores, logits = learner.predictor.predict(
            point_coords = np.array([learner.input_points[-1]]),
            point_labels = np.array([learner.input_labels[-1]]),
            mask_input = None,
            multimask_output = False,
            return_logits = True,
        )
        evidence = evidence.squeeze() #image format
        mask_from_seed = evidence > learner.predictor.model.mask_threshold
        learner.masks_from_single_seeds[mask_from_seed] = 1
    if len(learner.input_points) <= 1:
        print("should correct learner.masks_from_single_seeds when a false label is in a mask from a single seed")
    evidence, scores, learner.logits = learner.predictor.predict(
        point_coords = np.array(learner.input_points),
        point_labels = np.array(learner.input_labels),
        mask_input =  learner.logits if learner.use_previous_logits else None,
        multimask_output = False,
        return_logits = True,
    )
    learner.evidence = evidence.squeeze() #image format
    learner.cp_mask = np.bitwise_or(learner.evidence > learner.predictor.model.mask_threshold, learner.masks_from_single_seeds)
    