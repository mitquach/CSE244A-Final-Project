# CSE244A-Final-Project
This repository was made by Jonathan Casper, Daniel Sabo, and Michelle Quach for UCSC's CSE244A final project. It uses Meta's [Data Efficient Image Transformer](https://github.com/facebookresearch/deit) to classify dog and plant images provided from the Kaggle competition [here](https://www.kaggle.com/competitions/ucsc-cse-244-a-2024-fall-final-project/overview). 

## Requirements 
The dependencies for is project are specified by the provided `requirements.txt` file. Development was performed using python 3.11. Running the code via GPU is also highly recommended. 

```
conda create -n CSE244A_projectJDM python==3.11.10
conda activate CSE244A_projectJDM
pip install -r requirements.txt
```

## How to Run
Clone or download the entire GitHub repository and open "FinalProject.ipynb" Update the paths in the cell with a TODO. The 'data_prefix' path should be the parent directory of the train and test data. The 'model_prefix' path should be a subdirectory within the 'data_prefix' directory named 'models' which will contain a folder of all the checkpoint and history information for that specific model configuration. This should also be the parent of where the CSV files for the labels are too. Run all cells in a linear fashion. Once cells complete, a CSV file titled "test_submission.csv" will be generated. This file will have the image name and its predicted labels in it. 

## Example Directory Tree
```
ucsc-cse-244-a-2024-fall-final-project
    ├── models
    │   └── michelle_diet_imagenetmean_augment__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    ├── test
    └── train
        ├── labeled
        └── unlabeled
    ├── FinalProject.ipynb
    ├── README.md
    ├── michelle_diet_imagenetmean_augment__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4
    ├── requirements.txt
    └── test_submission.csv
```
