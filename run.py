from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

from models.LSTM import LSTM_
from models.LSTM_CNN import LSTM_CNN_
from models.CNN import CNN_
from models.GRU import GRU_
from models.base_model import base_model
from data_preprocessing.read_and_preprocess import *
from sklearn.model_selection import train_test_split
import numpy as np
import urllib.request
import inspect, os



full = True
glove_dimension = 300
max_words = 100
model_name = "LSTM"

# define paths for downloading our embedding and pre trained data
data_dir_name = "data/generated_gloved_%s_words_%s_full_%s" % (glove_dimension, max_words, full)
if not os.path.isdir(data_dir_name):
    os.mkdir(data_dir_name)
embedding_matrix_path = os.path.join(data_dir_name, "embedding_matrix.npy")
test_data_path = os.path.join(data_dir_name, "X_test_seq.npy")

model_save_dir = "generated_parameters/"+model_name
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)

checkpoint_path = os.path.join(model_save_dir, "best_checkpoint.hdf5")
model_json_path = os.path.join(model_save_dir, "model.json")

# load embedding matrix and test tweets
print("\nDownload and Load embedding matrix and X_test sequences...")
urllib.request.urlretrieve ("https://drive.switch.ch/index.php/s/6oIs5LPs39ebhfR/download",embedding_matrix_path)
embedding_matrix = np.load(embedding_matrix_path , allow_pickle=True)
urllib.request.urlretrieve ("https://drive.switch.ch/index.php/s/PRA84xnsHExNZtc/download",test_data_path)
X_test = np.load(test_data_path , allow_pickle=True)
print("Data Loaded...")
print("**********************************\n")

# download our pretrained data
print("Download model checkpoints...")
urllib.request.urlretrieve ("https://drive.switch.ch/index.php/s/nkATpktCHPlNlqd/download",checkpoint_path)
urllib.request.urlretrieve ("https://drive.switch.ch/index.php/s/EEyj6G8E82rBkeT/download",model_json_path)
print("Model downloaded...")
print("**********************************\n")
params = {
        'loss': 'binary_crossentropy',
        'num_neurons': 100,
        'dropout': 0.0,
        'batch_size': 512,
        'recurrent_dropout': 0.0,
        'epochs': 5,
        'dense_activation': 'sigmoid',
        'optimizer': 'RMSprop'
        }

modelobj = LSTM_(model_name , embedding_matrix , max_words , params)
print("++++++++++++++++++++++++++++++++++")
print(model_name)
modelobj.build_model()
modelobj.load()
print(modelobj.model.summary())
print("++++++++++++++++++++++++++++++++++")

print("-------generate labels for test data--------")
modelobj.prediction(X_test, model_name, batch_size=1024)
print("final .csv result is saved in results folder :) ")





