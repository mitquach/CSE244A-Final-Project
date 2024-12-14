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
### To reproduce the entire training procedure:
1. Clone the above repository.
2. Specify the path to the "ucsc-cse-244-a-2024-fall-final-project" data in the cell marked “#TODO”
3. Set evaluation_mode = False in the same cell
4. Run everything.
5. A file named 'test_submission*.csv' will be produced, each time the notebook is run a new prediction will be produced with an incremented suffix number.

### To reproduce the results of our pre-trained checkpoint:
1. Clone the above repository.
2. Download and extract the checkpoint zip file to the “models” directory within the repository.
3. Specify the path to the "ucsc-cse-244-a-2024-fall-final-project" data in the cell marked “#TODO”
4. Set evaluation_mode = True in the same cell
5. Run everything.
6. A file named 'test_submission*.csv' will be produced, each time the notebook is run a new prediction will be produced with an incremented suffix number.

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
