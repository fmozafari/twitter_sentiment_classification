
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

from models.LSTM import LSTM_
from models.CNN import CNN_
from data_preprocessing.read_and_preprocess import *

# import sys,os
# sys.path.insert(1,'data_preprocessing/')
# sys.path.insert(1,'models/')
# from read_and_preprocess import *
#from data_preprocessing.read_and_preprocess import *
from sklearn.model_selection import train_test_split
#from LSTM import LSTM_


full = True
glove_dimension = 25
max_words = 40

# load data
print("Loading Data...")
X_train, Y_train, X_test, embeding_matrix = load_data(FULL=full , GLOVE_DIMENSION=glove_dimension , MAX_WORDS=max_words)
print("Data Loaded...")

print(embeding_matrix.shape[0])

xtrain , xvalid, ytrain, yvalid = train_test_split(X_train, Y_train, shuffle=True, test_size=0.1, random_state=0) # for cross validation

# model parameters
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

model_name = "LSTM"
lstm = LSTM_(model_name , embeding_matrix , max_words)
lstm.build_model(params)
print("----model build----- ")
print("----model summary----")
print(lstm.model.summary())
# Train the model ---> save the weights with best validation loss
print("----training.....")
lstm.train(xtrain, ytrain, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(xvalid, yvalid))
print("------model trained------")
print("++++++++++++++++++++++++++++++++++")
print(model_name)
print("++++++++++++++++++++++++++++++++++")

print("-------generate labels for test data--------")
lstm.load()
print(lstm.model.summary())
xtest = X_test
lstm.prediction(xtest, model_name, batch_size=1024)

# model_name = "CNN"
# cnn = CNN_(model_name , embeding_matrix , max_words)

# cnn.load()
# print(cnn.model.summary())
# xtest = X_test
# cnn.prediction(xtest, model_name, batch_size=1024)




