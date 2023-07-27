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

This project implements a flexible pipeline to code easily any active learning strategy for segmentation.The pipeline contains a number of modules that can be used to code a large number of different strategies.

- 4 Learner types are available: 
    - ActiveLearningSAM: This is the fundamental class for the active Learning for Connectomics project. This class should be used for every active learner that need not access ground truth.
    - PseudoActiveLearningSAM: This class is an active learner that can access the ground truth. It allows simulating specific settings that need to use the ground truth. For instance, providing all foreground points at the beginning can only be done in this class.
    - FPFNLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is based on false positive and false negative pixels.
    - RandomLearner: This class is not an active learner, it has been implemented to compare active learners to other type of learners. In this class, sampling is done randomly. 

## How to use the pipeline ?
### With AL_SAM


### With run_model


## Contributors

This project was made possible with the help of many contributors:

Clément Yvernes, Donglai Wei, Junsik Kim and Yifan (Rosetta) Hu.

