import numpy as np
import warnings
from tools import *
from plot_tools import *


class BasicLS:
    def __init__(self, learner) -> None:
        self.learner = learner
        self.use_dropout = False
    
    """
    Useful decorators
    """
    def apply_dropout_parameters(func):
        def wrapper(self):
            self.learner.predictor.use_dropout = self.use_dropout
            func(self)
            self.learner.predictor.use_dropout = False 
        return(wrapper)
    

    @apply_dropout_parameters
    def __call__(self) ->None:

        """
        Segmentaing and Estimating Uncertainties all at once
        """
        evidence, scores, self.learner.logits = self.learner.predictor.predict(
            point_coords=np.array(self.learner.input_points),
            point_labels=np.array(self.learner.input_labels),
            mask_input=self.learner.logits if self.learner.use_previous_logits else None,
            multimask_output=False,
            return_logits=True,
        )
        self.learner.evidence = evidence.squeeze()  # image format

        """
        Final prediction
        """
        self.learner.cp_mask = self.learner.evidence > self.learner.predictor.model.mask_threshold


class OneSeedForOneMaskLS(BasicLS):
    
    def __init__(self, learner) -> None:
        self.learner = learner
        self.use_dropout = False
    
    def segmentation_strategy(self) -> None:
        for idx in range(self.learner.nb_seed_used, len(self.learner.input_points)):
            if self.learner.input_labels[idx]:
                evidence, scores, logits = self.learner.predictor.predict(
                    point_coords=np.array([self.learner.input_points[idx]]),
                    point_labels=np.array([self.learner.input_labels[idx]]),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                )
                evidence = evidence.squeeze()  # image format
                mask_from_seed = evidence > self.learner.predictor.model.mask_threshold
                self.learner.masks_from_single_seeds [mask_from_seed] = 1
        
    def uncertainty_strategy(self) -> None:
        evidence, scores, self.learner.logits = self.learner.predictor.predict(
            point_coords=np.array(self.learner.input_points),
            point_labels=np.array(self.learner.input_labels),
            mask_input=self.learner.logits if self.learner.use_previous_logits else None,
            multimask_output=False,
            return_logits=True,
        )
        self.learner.evidence = evidence.squeeze()  # image format
    
    def do_final_prediction(self) -> None:
        self.learner.cp_mask = np.bitwise_or(
            self.learner.evidence > self.learner.predictor.model.mask_threshold,
            self.learner.masks_from_single_seeds  > 0,
        )
    
    @BasicLS.apply_dropout_parameters
    def __call__(self) -> None:
        self.segmentation_strategy()
        self.uncertainty_strategy()
        self.do_final_prediction()


class OneSeedForOneMaskLSWithDropOut(OneSeedForOneMaskLS):
    def __init__(self, learner) -> None:
        super().__init__(learner)
        self.use_dropout = True
        self.learner.predictor.setDropOutParameter(0.3)
        self.nb_tests = 20
    
    def uncertainty_strategy(self) -> None:
        
        evidence = np.zeros_like(self.learner.evidence)
        if self.learner.use_previous_logits:
            logits = np.zeros_like(self.learner.logits)
            
        for idx in range(self.nb_tests):
            current_evidence, current_scores, current_logits = self.learner.predictor.predict(
                point_coords=np.array(self.learner.input_points),
                point_labels=np.array(self.learner.input_labels),
                mask_input=self.learner.logits/(idx + 1) if self.learner.use_previous_logits else None,
                multimask_output=False,
                return_logits=True,
            )
            evidence += current_evidence.squeeze()  # image format
            if self.learner.use_previous_logits:
                logits += current_logits
                
        self.learner.evidence = evidence / self.nb_tests
        if self.learner.use_previous_logits:
            self.learner.logits = logits / self.nb_tests

        
    

class FewSeedsForOneMaskLS(OneSeedForOneMaskLS):
    def __init__(self, learner) -> None:
        super().__init__(learner)

    def segmenting_strategy(self) -> None:
        for idx in range(self.learner.nb_seed_used, len(self.learner.input_points)):
            # Computing the object around this seed
            seed = self.learner.input_points[idx]
            evidence_temp, _, _ = self.learner.predictor.predict(
                point_coords=np.array([seed]),
                point_labels=np.array([True]),
                mask_input=None,
                multimask_output=False,
                return_logits=True,
            )
            evidence_temp = evidence_temp.squeeze()  # image format
            mask_temp = evidence_temp > self.learner.predictor.model.mask_threshold
            previous_prediction = self.learner.masks_from_single_seeds  [seed[1], seed[0]] 
            if previous_prediction > 0:
                print(self.learner.seed_subsets, previous_prediction - 1)
                #visual_center = findVisualCenter(intersection_of_segmentation)
                #value_at_visual_center = self.learner.masks_from_single_seeds  [visual_center[0], visual_center[1]]
                corresponding_subset = abs(previous_prediction) - 1
                self.learner.seed_subsets[corresponding_subset].append(self.learner.input_points[idx])
                self.learner.label_subsets[corresponding_subset].append(self.learner.input_labels[idx])
                new_segmentation, _, _ = self.learner.predictor.predict(
                    point_coords=np.array(self.learner.seed_subsets[corresponding_subset]),
                    point_labels=np.array(self.learner.label_subsets[corresponding_subset]),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                )
                new_segmentation = new_segmentation.squeeze()
                blob_to_change = self.learner.masks_from_single_seeds   == previous_prediction
                self.learner.masks_from_single_seeds  [blob_to_change] = new_segmentation[blob_to_change]

            else:
                self.learner.masks_from_single_seeds  [np.logical_and(mask_temp, self.learner.masks_from_single_seeds   <= 0)] = (
                    (len(self.learner.seed_subsets) + 1) if self.learner.input_labels[idx] else -(len(self.learner.seed_subsets) + 1)
                )
                self.learner.seed_subsets.append([self.learner.input_points[idx]])
                self.learner.label_subsets.append([self.learner.input_labels[idx]])

    def do_final_prediction(self) -> None:
       self.learner.cp_mask = self.learner.masks_from_single_seeds  > 0
