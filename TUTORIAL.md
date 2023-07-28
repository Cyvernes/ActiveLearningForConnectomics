# Tutorial

## Description of the pipeline

This project implements a flexible pipeline to easily code many active learning strategies for segmentation. It contains a large number of modules that allow you to code a very large number of learning strategies. Each strategy is built by assembling elements of each module. It is therefore necessary to understand the interest of each module.This paragraph provides a short description of each module. Should further details be required, [documentation](https://cyvernes.github.io/AL_Docs/index.html#welcome-to-active-learning-for-connectomics-s-documentation) is available.

- Four [Learners](https://cyvernes.github.io/AL_Docs/Learners.html#module-Learners) are available. Learners are the main classes in the pipeline, they contain all the methods called during an active learning process: 
    - ActiveLearningSAM: This is the fundamental class for the active Learning for Connectomics project. This class should be used for every active learner that need not access ground truth.

    - PseudoActiveLearningSAM: This class is an active learner that can access the ground truth. It allows simulating specific settings that need to use the ground truth. For instance, providing all foreground points at the beginning can only be done in this class.

    - FPFNLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is based on false positive and false negative pixels.

    - RandomLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is done randomly. 
    
- Many [learning strategies](https://cyvernes.github.io/AL_Docs/learning_strategies.html#module-learning_strategies) are available. Learning strategies update their Learner’s evidence map and current predicted mask. The learning strategies are divided into three distinct sub-strategies:
    - A segmentation strategy that aims to have the best possible segmentation for already annotated objects.
    - An uncertainty strategy that aims to use every annotated points to estimate region of uncertainty and the evidence map (logits).
    - The final prediction strategy that uses the results of the segmentation strategy and the uncertainty strategy to have a better segmentation.
PLease refer to the [documentation](https://cyvernes.github.io/AL_Docs/learning_strategies.html#module-learning_strategies) to have a description of all learning strategies natively available.

- Some [sampling strategies](https://cyvernes.github.io/AL_Docs/next_seeds_strategies.html#module-next_seeds_strategies) are available. Sampling strategies use the evidence map or the uncertainty map to sample the next points to annotate. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/next_seeds_strategies.html#module-next_seeds_strategies) to have a description of all sampling strategies natively available.

- Some [filters](https://cyvernes.github.io/AL_Docs/filters.html#module-filters) are available. Filters are functions that modify values of an array. They are used to solve problems encountered in naive implementations of sampling strategies. Usually filters are used in a specific region of interest. These regions are computed thanks to auxiliary functions. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/filters.html#module-filters) to have a description of all filters and auxiliary functions natively available.

- Some [strategy selectors](https://cyvernes.github.io/AL_Docs/strategy_selectors.html#module-strategy_selectors) are available. These function computes which learning or sampling strategy should be used at each moment of the learning procedure. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/strategy_selectors.html) to have a description of all strategy selectors natively available.

- Some [strategies to sample the first seeds](https://cyvernes.github.io/AL_Docs/first_seeds_selector.html#module-first_seeds_selector) are available. They decide which points should be annotated in first. Please note that some of these strategies can be used to simulate specific settings, they might require to access the ground truth. In this case, PseudoActiveLearningSAM should be used instead of ActiveLearningSAM.


- Additional tools such as [data tools](https://cyvernes.github.io/AL_Docs/data_tools.html#module-data_tools), [Visualisation tools](https://cyvernes.github.io/AL_Docs/plot_tools.html#module-plot_tools) and [miscellaneous tools](https://cyvernes.github.io/AL_Docs/tools.html#module-tools) are available.

- Two scripts, [AL_SAM](https://cyvernes.github.io/AL_Docs/AL_SAM.html#module-AL_SAM) and [run_model](https://cyvernes.github.io/AL_Docs/run_model.html#module-run_model) are available to start experiments easily.

## How to use the pipeline

### For Single Image

To quickly launch an experiment on a single image, it is recommended to use AL_SAM. It is possible to use AL_SAM to run experiments on a dataset but the results will not be aggregated. That's why it's recommended to use run_model to run experiments on a dataset.

