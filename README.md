# Active Learning for Connectomics

This project applies Active Learning procedures to the [SAM model](https://github.com/facebookresearch/segment-anything) for performing mitochondrial segmentation with point annotation.

## Installation


We strongly recommend creating a separate conda environment:

```
conda create -n <environment_name> python=3.10
conda activate <environment_name>
```

To begin with, clone the repository to your local machine:

```
git clone https://github.com/Cyvernes/ActiveLearningForConnectomics.git
```

After activating the environment, navigate to the project directory and install the necessary packages:

You can try to install necessary packages with the requirements.txt

```
cd ActiveLearningForConnectomics
pip install -r requirements.txt
```
If these requirements does not work on your machine, please follow the instalation procedure of the [official SAM Repository](https://github.com/facebookresearch/segment-anything#installation)

This projects uses SAM and our experiments have been done with the [ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) SAM model checkpoint. It is recomended to use this checkpoint. Should your machine not able to load the checkpoint, other checkpoints are available on the [official SAM Repository](https://github.com/facebookresearch/segment-anything#installation).

## Getting Started

To get started with the pipeline, please follow the [tutorial](https://github.com/Cyvernes/ActiveLearningForConnectomics/blob/main/TUTORIAL.md)


## Pipeline Features

The project offers a flexible pipeline for coding various active learning strategies for segmentation. You can find more details in the [tutorial](https://github.com/Cyvernes/ActiveLearningForConnectomics/blob/main/TUTORIAL.md) or in the [documentation](https://cyvernes.github.io/AL_Docs/index.html#welcome-to-active-learning-for-connectomics-s-documentation).

Key components of the pipeline include:

- Four types of [Learners](https://cyvernes.github.io/AL_Docs/Learners.html#module-Learners)
- A variety of [learning strategies](https://cyvernes.github.io/AL_Docs/learning_strategies.html#module-learning_strategies)
- Various options for [initial seed sampling](https://cyvernes.github.io/AL_Docs/first_seeds_selector.html#module-first_seeds_selector)
- Several [sampling strategies](https://cyvernes.github.io/AL_Docs/next_seeds_strategies.html#module-next_seeds_strategies)
- Different [strategy selectors](https://cyvernes.github.io/AL_Docs/strategy_selectors.html#module-strategy_selectors)
- Multiple [filters](https://cyvernes.github.io/AL_Docs/filters.html#module-filters)
- Additional tools, such as [data tools](https://cyvernes.github.io/AL_Docs/data_tools.html#module-data_tools), [visualization tools](https://cyvernes.github.io/AL_Docs/plot_tools.html#module-plot_tools), and [miscellaneous tools](https://cyvernes.github.io/AL_Docs/tools.html#module-tools)

## Contributors

This project represents a collaborative effort with valuable contributions from many individuals. They are listed in alphabetical order:

Clément Yvernes, Donglai Wei, Junsik Kim, and Yifan (Rosetta) Hu.
