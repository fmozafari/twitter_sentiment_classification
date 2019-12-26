# Twitter Sentiment Classification
This project is created as part of the Machine Learning course [CS-433] at EPFL. We developed a tweet sentiment classfication using some state-of-the-art Neural Network models.

## crowdAI last submission
- ID: 30300
- Username: fereshte
- Time: Tue, 17 Dec 2019, 17:37:16

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

## Regenerate our best final result
To regenerate our final result from pre generated embedding data and pre trained model, you only need to execute "python run.py"
This file download our embedding data for 300 dimension and our best model checkpoints (for LSTM). Afterwards, predict labels for test tweets data. The final .csv result is saved in results folder.

## Dataset
The dataset is available from "https://nlp.stanford.edu/projects/glove/" page. Download the Twitter datasets (glove.twitter.27B.zip) and Common Crawl (glove.840B.300d.zip)
- Extract zip folder and put glove files in "data/glove_embeddings" path
- Put twitter datasets including train pos, train neg, and test tweets in "data/twitter-dataset" path

## Setup
To run for every data and model. First, locate datasets in data folder. You can use train_predict.py to extract your own results. Next, specify {full, glove_dimension, max_words, model_name} parameters and then execute "python train_predict.py". Final results will be saved in results directory.

## Folder structure of project
```
.
├── baseline_models        # This folder contains the implementation of RLR and SVM models as our baselines
├── data                   # This folder includes twitter datasets and glove dataset
├── data_preprocessing     # Functions for reading data, preprocessing, generating data from glove embedding matrix, and loading data
├── generated_parameters   # This folder includes generated checkpoints from training of models
├── models                 # This folder includes the implementation of neural network models
├── paths                  # json file to specify relative paths
├── results                # This folder consists of .csv result file for submitting to crowdAI
├── README.md              # README file
├── run.py                 # Script for running the best model and creating result file
└── train_predict.py       # This file generate and load data, train the desired model, and finally generate prediction for test data
```

## Training models
For this classification project, we have implemented 3 neural network models:
- LSTM
- CNN
- GRU


