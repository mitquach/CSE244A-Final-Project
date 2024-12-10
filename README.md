# CSE244A-Final-Project
This repository was made by Jonathan Capser, Daniel Sabo, and Michelle Quach for UCSC's CSE244A final project. It uses Meta's Data Efficient Image Transformer to classify dog and plant images provided from the Kaggle competition [here](https://www.kaggle.com/competitions/ucsc-cse-244-a-2024-fall-final-project/overview). 

## Requirements 
Use the requirements.txt file to install all the proper dependencies for this repository. Running the code via GPU is also highly recommended. 

## How to Run
Clone or download the entire GitHub repository and open "FinalProject.ipynb" Update the paths in the cell with a TODO. The path should be the parent directory of the train and test data. This should also be the parent of where the CSV files for the labels are too. Run all cells in a linear fashion. Once cells complete, a CSV file titled "test_submission.csv" will be generated. This file will have the image name and its predicted labels in it. 

## Example Directory Tree
```
ucsc-cse-244-a-2024-fall-final-project
    ├── models
    │   ├── michelle_diet384_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    │   ├── michelle_diet_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    │   ├── michelle_distilled_diet_imagenetmean_augmentD__freeze11__cosine_1en5_0.8__AdamW_wdecay_1en3
    │   ├── michelle_distilled_diet_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    │   ├── michelle_distilled_diet_imagenetmean_augment__freeze11__cosine_1en3_0.8__AdamW_wdecay_1en4
    │   └── michelle_distilled_diet_imagenetmean_augment__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    ├── test
    └── train
        ├── labeled
        └── unlabeled
    ├── FinalProject.ipynb
    ├── FinalProject_kfold.ipynb
    ├── README.md
    ├── michelle_diet384_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_diet__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4.yaml
    ├── michelle_diet__freeze11__plateaulr_0.1_0_0.0__wdecay_1en4.yaml
    ├── michelle_diet__plateaulr_0.1_0_0.0.yaml
    ├── michelle_diet__steplr_3_0.97.yaml
    ├── michelle_diet_imagenetmean__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4.yaml
    ├── michelle_diet_imagenetmean_augmentD__freeze10__explr_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_diet_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_diet_imagenetmean_augmentD__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4.yaml
    ├── michelle_diet_imagenetmean_augment__freeze11__cosine_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_distilled_diet_imagenetmean_augmentD__freeze11__cosine_1en5_0.8__AdamW_wdecay_1en3.yaml
    ├── michelle_distilled_diet_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_distilled_diet_imagenetmean_augment__freeze11__cosine_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── michelle_distilled_diet_imagenetmean_augment__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4.yaml
    ├── requirements.txt
    └── test_submission.csv
```