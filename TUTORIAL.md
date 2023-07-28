# Tutorial

## Description of the pipeline

This project implements a flexible pipeline to code easily any active learning strategy for segmentation.The pipeline contains a number of modules that can be used to code a large number of different learning strategies. For more details a [documentation](https://cyvernes.github.io/AL_Docs/index.html#welcome-to-active-learning-for-connectomics-s-documentation) is available.

- Four [Learners](https://cyvernes.github.io/AL_Docs/Learners.html#module-Learners) are available: 
    - ActiveLearningSAM: This is the fundamental class for the active Learning for Connectomics project. This class should be used for every active learner that need not access ground truth.

    - PseudoActiveLearningSAM: This class is an active learner that can access the ground truth. It allows simulating specific settings that need to use the ground truth. For instance, providing all foreground points at the beginning can only be done in this class.

    - FPFNLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is based on false positive and false negative pixels.

    - RandomLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is done randomly. 
    
- Many [learning strategies](https://cyvernes.github.io/AL_Docs/learning_strategies.html#module-learning_strategies) are available. Each learning strategy implements a segmentation stratey, an uncertainty strategy and a final_prediction strategy. They update the Learner’s evidence map and current predicted masks using corresponding strategies. Each Learning strategy is made of three components:
    - A segmentation strategy that aims to have the best possible segmentation for already annotated objects.
    - An uncertainty strategy that aims to use every annotated points to estimate region of uncertainty and the evidence map (logits).
    - The final prediction strategy that uses the results of the segmentation strategy and the uncertainty strategy to have a better segmentation.
PLease refer to the [documentation](https://cyvernes.github.io/AL_Docs/learning_strategies.html#module-learning_strategies) to have a description of all learning strategies natively available.

- Some [sampling strategies](https://cyvernes.github.io/AL_Docs/next_seeds_strategies.html#module-next_seeds_strategies) are available. Sampling strategies use the evidence map or the uncertainty map to sample the next points to annotate. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/next_seeds_strategies.html#module-next_seeds_strategies) to have a description of all sampling strategies natively available.

- Some [filters](https://cyvernes.github.io/AL_Docs/filters.html#module-filters) are available. Filters are functions that modify values of an array. They are used to solve problems encountered in naive implementations of sampling strategies. Usually filters are used in a specific region of interest. These region are computed thanks to auxiliary functions. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/filters.html#module-filters) to have a description of all filters and auxiliary functions available.

- Some [strategy selectors](https://cyvernes.github.io/AL_Docs/strategy_selectors.html#module-strategy_selectors) are available. These function computes which learning or sampling strategy should be used at each moment of the learning procedure. Please refer to the [documentation](https://cyvernes.github.io/AL_Docs/strategy_selectors.html) to have a description of all strategy selectors natively available.

- Some [strategies to sample the first seeds](https://cyvernes.github.io/AL_Docs/first_seeds_selector.html#module-first_seeds_selector) are available. They decide which points should be annotated in first. Please note that some of these strategies can be used to similate specific setting, they might require to access the ground truth. In this case, PseudoActiveLearningSAM should be used instead of ActiveLearningSAM.


- Many tools are available. They are used by every module of the pipeline. They provide [data tools](https://cyvernes.github.io/AL_Docs/data_tools.html#module-data_tools), [Visualisation tools](https://cyvernes.github.io/AL_Docs/plot_tools.html#module-plot_tools) and [miscellaneous tools](https://cyvernes.github.io/AL_Docs/tools.html#module-tools)