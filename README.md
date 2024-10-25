# XAI-integrated Visual Quality Inspection

## Introduction

This repository contains the code for the paper "XEdgeAI: A Human-centered Industrial Inspection Framework with
Data-centric Explainable Edge AI Approach" submitted to Information Fusion/Elsevier.

## Requirements

Install the required libraries using the following command:

```
pip install -r requirements.txt
```

## Usage

The code is written in Python and is based on the PyTorch library.
The code is used to train and evaluate the proposed XAI-integrated Visual Quality Inspection Framework. The code is
organized as follows:

- `data/`: contains the dataset used in the paper.
- `model/`: contains the implementation of the proposed XAI-integrated Visual Quality Inspection Framework.
    - `split_dataset.py`: splits the dataset into training and testing sets.
    - `retrain.py`: finetune the model.
    - `evaluate.py`: evaluate the model.
    - `mobile_opt.py`: optimize the model for deployment.
- `explainer/`: contains the implementation of the proposed XAI methods used in the paper.
    - `cam.py`: explain the model using the CAM methods.
    - `rise.py`: explain the model using the RISE method.
    - `metrics.py`: calculate the plausibility and faithfulness of the explanations.
- `logs/`: contains the logs of the training process.
- `requirements.txt`: contains the required libraries.

## Citation
If you find this code useful, please consider citing our paper :) Thank you!
```
@article{nguyen2024xedgeai,
  title={XEdgeAI: A Human-centered Industrial Inspection Framework with Data-centric Explainable Edge AI Approach},
  author={Nguyen, Truong Thanh Hung and Nguyen, Phuc Truong Loc and Cao, Hung},
  journal={arXiv preprint arXiv:2407.11771},
  year={2024}
}
```
