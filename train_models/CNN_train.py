
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
from CNN import CNN_

full = True
glove_dimension = 25
max_words = 40

# load data
print("Loading Data...")
X_train, Y_train, X_test, embeding_matrix = load_data(FULL=full , GLOVE_DIMENSION=glove_dimension , MAX_WORDS=max_words)
print("Data Loaded...")

print(embeding_matrix.shape[0])

#train , xvalid, ytrain, yvalid = train_test_split(X_train, Y_train, shuffle=True, test_size=0.1, random_state=0) # for cross validation

# model parameters
params = {
    'filters': 80,
    'kernel_size': 10,
    'activation': "relu",
    'MP_pool_size': 2,
    'epochs': 5,
    'batch_size': 1024,
    'dense_activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'optimizer': 'RMSprop'
}


model_name = "CNN"
cnn = CNN_(model_name , embeding_matrix , max_words)
cnn.build_model(params)
print("----model build----- ")
print("----model summary----")
print(cnn.model.summary())
# Train the model ---> save the weights with best validation loss
print("----training.....")
cnn.train(X_train, Y_train, epochs=params["epochs"], batch_size=params["batch_size"])
print("------model trained------")
print("++++++++++++++++++++++++++++++++++")
print(model_name)
print("++++++++++++++++++++++++++++++++++")

