# Active Learning For Connectomics

This project aims to use Active Learning procedures on the [SAM model](https://github.com/facebookresearch/segment-anything) to perform mitochondrial segmentation with point annotation.

## Instalation

The instalation works well with python 3.10. Please follow the instruction of the [SAM repository](https://github.com/facebookresearch/segment-anything#installation) to install SAM correctly. It is strongly recomended to do the installation in a new python environment.

Please install OpenCV and other depencies of the project.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Clone the repository

```
git clone git@github.com:Cyvernes/ActiveLearningForConnectomics.git
```

## Getting Started

SAM needs a [model checkpoint](git@github.com:Cyvernes/ActiveLearningForConnectomics.git) to work properly. This project have been devellopped with the ViT-H SAM model, you can use this [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to download it.

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

- Two scripts, [AL_SAM](https://cyvernes.github.io/AL_Docs/AL_SAM.html#module-AL_SAM) and [run_model](https://cyvernes.github.io/AL_Docs/run_model.html#module-run_model) are available to start experiments easily.
## How to use the pipeline ?
### With AL_SAM
AL_SAM is a script made to run experiments on single images, results are not aggregated. At the beginning of the script some parameters have to be selected. 
- Learners parameters: They define the learning strategy to test. One can choose the type of the learner and every other parameters

- Budget parameters: They define the maximum number of annotation authorized for every image. If USE_BUDGET is set to False, the budget would be the number of seeds computed by the first sampling strategy.

- Plots and results parameters: They define which visualization should be displayed throughout the experiment.

- Data parameters: They define the dataset on which the experiment is run. 

### With run_model


## Contributors

This project was made possible with the help of many contributors. They are listed in alphabetical order:

Clément Yvernes, Donglai Wei, Junsik Kim and Yifan (Rosetta) Hu.

