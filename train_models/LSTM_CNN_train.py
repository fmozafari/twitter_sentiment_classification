
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

import sys,os
sys.path.insert(1,'data_preprocessing/')
sys.path.insert(1,'models/')
from read_and_preprocess import *
#from data_preprocessing.read_and_preprocess import *
from sklearn.model_selection import train_test_split
from LSTM_CNN import LSTM_CNN

full = True
glove_dimension = 25
max_words = 40

# load data
print("Loading Data...")
X_train, Y_train, X_test, embeding_matrix = load_data(FULL=full , GLOVE_DIMENSION=glove_dimension , MAX_WORDS=max_words)
print("Data Loaded...")

xtrain , xvalid, ytrain, yvalid = train_test_split(X_train, Y_train, shuffle=True, test_size=0.1, random_state=0) # for cross validation

# model parameters
params = {
    'LSTM_num_neurons': 150,
    'LSTM_dropout': 0,
    'LSTM_recurrent_dropout': 0,
    'CNN_filters': 128,
    'CNN_kernel_size': 5,
    'CNN_activation': "relu",
    'CNN_pool_size': 2,
    'epochs': 1,
    'batch_size': 1024,
    'DENSE_activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'optimizer': 'RMSprop'
}


model_name = "LSTM_CNN"
lstm_cnn = LSTM_CNN(model_name , embeding_matrix , max_words , params)
lstm_cnn.build_model()
print("----model build----- ")
print("----model summary----")
print(lstm_cnn.model.summary())
# Train the model ---> save the weights with best validation loss
print("----training.....")
lstm_cnn.train(xtrain, ytrain, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(xvalid, yvalid))
print("------model trained------")
print("++++++++++++++++++++++++++++++++++")
print(model_name)
print("++++++++++++++++++++++++++++++++++")

