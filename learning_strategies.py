import numpy as np
import warnings
from tools import *
from plot_tools import *
from Learners import ActiveLearningSAM


class BasicLS:
    """This class is the basic class for all learning strategy.
    Update the Learner’s evidence map and current predicted masks by corresponding strategies
    Each learning strategy implements a segmentation stratey, an uncertainty strategy and a final_prediction strategy.
    The segmentation strategy aims to have the best possible segmentation for already annotated objects.
    The uncertainty strategy aims to use every annotated points to estimate region of high uncertainty.
    The final prediction strategy use the results of the segmentation strategy and the uncertainty strategy to have a better segmentation.
    
    """
    def __init__(self, learner : ActiveLearningSAM) -> None:
        """Constructor of the class

        :param learner: Learner
        :type learner: ActiveLearningSAM
        """
        self.learner = learner
        self.use_dropout = False

    def apply_dropout_parameters(func):
        """This is a decorator to apply parameters locally when calling the object

        """
        def wrapper(self):
            self.learner.predictor.use_dropout = self.use_dropout
            func(self)
            self.learner.predictor.use_dropout = False

        return wrapper

    def segmentation_strategy(self) -> None:
        """There is no segmentation strategy in the basic learning strategy
        """
        return

    def uncertainty_strategy(self) -> None:
        """Computes the uncertainty using the evidence map from the SAM predictor
        """
        evidence, scores, self.learner.logits = self.learner.predictor.predict(
            point_coords=np.array(self.learner.input_points),
            point_labels=np.array(self.learner.input_labels),
            mask_input=self.learner.logits
            if self.learner.use_previous_logits
            else None,
            multimask_output=False,
            return_logits=True,
        )
        self.learner.evidence = evidence.squeeze()  # image format


    def do_final_prediction(self) -> None:
        """Do prediction in the same way as the SAM predictor
        """
        self.learner.cp_mask = (
            self.learner.evidence > self.learner.predictor.model.mask_threshold
        )
    
    @apply_dropout_parameters
    def __call__(self) -> None:
        """Method that call every strategy
        """
        self.segmentation_strategy()
        self.uncertainty_strategy()
        self.do_final_prediction()
    



class OneSeedForOneMaskLS(BasicLS):
    """SAM is better to segment single and isolated object.
        This learning strategy implements a segmentation strategy that take advantage of this observation.
    """
    def __init__(self, learner : ActiveLearningSAM) -> None:
        """Constructor of the class

        :param learner: Learner
        :type learner: ActiveLearningSAM
        """
        self.learner = learner
        self.use_dropout = False

    def segmentation_strategy(self) -> None:
        """SAM predictor is used on every foreground  point to have a good segmentation of the corresponding object. 
        This segmentation is never changed during the experiment. 
        """
        for idx in range(self.learner.nb_seed_used, len(self.learner.input_points)):
            if self.learner.input_labels[idx]:
                evidence, scores, logits = self.learner.predictor.predict(
                    point_coords=np.array([self.learner.input_points[idx]]),
                    point_labels=np.array([True]),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=True,
                )
                evidence = evidence.squeeze()  # image format
                mask_from_seed = evidence > self.learner.predictor.model.mask_threshold
                self.learner.masks_from_single_seeds[mask_from_seed] = 1


    def do_final_prediction(self) -> None:
        """Segmentation from the segmentation strategy is added to the segmentation from the uncertainty strategy
        """
        self.learner.cp_mask = np.bitwise_or(
            self.learner.evidence > self.learner.predictor.model.mask_threshold,
            self.learner.masks_from_single_seeds > 0,
        )



class OneSeedForOneMaskLSWithDropOut(OneSeedForOneMaskLS):
    """SAM is better to segment single and isolated object.
        This learning strategy implements a segmentation strategy that take advantage of this observation.
        This strategy uses drop out to reduce the overconfidence of the model.

    """
    def __init__(self, learner : ActiveLearningSAM) -> None:
        super().__init__(learner)
        self.use_dropout = True
        self.learner.predictor.setDropOutParameter(0.3)
        self.nb_tests = 20

    def uncertainty_strategy(self) -> None:
        """This uncertainty strategy is very similare to the one of OneSeedForOneMask except the uncertainty is computed several times with drop out
        The final evidence map is the pixel wise average of the differrent evidence map.
        """
        evidence = np.zeros_like(self.learner.evidence)
        if self.learner.use_previous_logits:
            logits = np.zeros_like(self.learner.logits)

        for idx in range(self.nb_tests):
            (
                current_evidence,
                current_scores,
                current_logits,
            ) = self.learner.predictor.predict(
                point_coords=np.array(self.learner.input_points),
                point_labels=np.array(self.learner.input_labels),
                mask_input=self.learner.logits / (idx + 1)
                if self.learner.use_previous_logits
                else None,
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
    """This learning strategy is similar to OneSeedForOneMaskLS but segmentation errors can be fixed by the annoatation.

    """
    def __init__(self, learner : ActiveLearningSAM) -> None:
        """Contructor of the class

        """
        super().__init__(learner)

    def segmentation_strategy(self) -> None:
        """This segmentation strategy allows to fix segmentation errors. If a previous segmentation from a single point was wrong. 
        A background point can be added to correct the error. The segmentation of the corresponding object would be done by the two seeds instead of a single one.
        """
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
            previous_prediction = self.learner.masks_from_single_seeds[seed[1], seed[0]]
            if previous_prediction > 0:
                # visual_center = findVisualCenter(intersection_of_segmentation)
                # value_at_visual_center = self.learner.masks_from_single_seeds  [visual_center[0], visual_center[1]]
                corresponding_subset = abs(previous_prediction) - 1
                self.learner.seed_subsets[corresponding_subset].append(
                    self.learner.input_points[idx]
                )
                self.learner.label_subsets[corresponding_subset].append(
                    self.learner.input_labels[idx]
                )
                new_segmentation, _, _ = self.learner.predictor.predict(
                    point_coords=np.array(
                        self.learner.seed_subsets[corresponding_subset]
                    ),
                    point_labels=np.array(
                        self.learner.label_subsets[corresponding_subset]
                    ),
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                )
                new_segmentation = new_segmentation.squeeze()
                blob_to_change = (
                    self.learner.masks_from_single_seeds == previous_prediction
                )
                self.learner.masks_from_single_seeds[blob_to_change] = new_segmentation[
                    blob_to_change
                ]

            else:
                self.learner.masks_from_single_seeds[
                    np.logical_and(mask_temp, self.learner.masks_from_single_seeds <= 0)
                ] = (
                    (len(self.learner.seed_subsets) + 1)
                    if self.learner.input_labels[idx]
                    else -(len(self.learner.seed_subsets) + 1)
                )
                self.learner.seed_subsets.append([self.learner.input_points[idx]])
                self.learner.label_subsets.append([self.learner.input_labels[idx]])

    def do_final_prediction(self) -> None:
        """Only the mask from the segmentation strategy is used in this learning strategy.
        """
        self.learner.cp_mask = self.learner.masks_from_single_seeds > 0
