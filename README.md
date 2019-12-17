# Twitter Sentiment Classification
This project is created as part of the Machine Learning course [CS-433] at EPFL. We developed a tweet sentiment classfication using some state-of-the-art Neural Network models.

## Dataset
The dataset is available from "https://nlp.stanford.edu/projects/glove/" page. Download the Twitter datasets (glove.twitter.27B.zip) and Common Crawl (glove.840B.300d.zip)
- Extract zip folder and put glove files in "data/glove_embeddings" path
- Put twitter datasets including train pos, train neg, and test tweets in "data/twitter-dataset" path

## crowdAI last submission
- ID:
- Username:
- Link:

## Used Libraries
```
- Python 3.6.5
- numpy 1.15.4
- pandas 0.23.4
- Tensorflow 1.4.1
- keras 2.0.8
- scikit-learn 0.20.1
- h5py 2.8.0
```

## Setup
Run "python run.py" simply
Note that run file execute classification with best model (LSTM) to extract the results

## Folder structure of project
.
├── data                   # This folder includes twitter datasets and glove dataset
├── data_preprocessing     # Functions for reading data, preprocessing, generating data from glove embedding matrix, and loading data
├── generated_parameters   # This folder includes generated checkpoints from training of models
├── models                 # This folder includes the implementation of baseline model and neural network models
├── paths                  # json file to specify relative paths
├── results                # This folder consists of .csv result file for submitting to crowdAI
├── train_models           # This folder includes all scripts for training different models
├── README.md              # README file
└── run.py                 # Script for running the best model and creating result file

## Training models
For this classification project, we have implemented 4 neural network models:
- LSTM
- CNN
- LSTM-CNN
- GRU 

## Regenerate our final result

## Authors
- Fereshte Mozafari   fereshte.mozafari@epfl.ch
- Mohammad Vahdat     mohammad.vahdat@epfl.ch
- Ehsan Mohammadpour  
