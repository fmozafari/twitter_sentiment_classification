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


full = True
glove_dimension = 25
max_words = 40
model_name = "LSTM"

# load data
print("Loading Data...")
X_train, Y_train, X_test, embeding_matrix = load_data(FULL=full , GLOVE_DIMENSION=glove_dimension , MAX_WORDS=max_words)
print("Data Loaded...")

#  model parameters
params = {
    'LSTM' : 
        {
        'loss': 'binary_crossentropy',
        'num_neurons': 100,
        'dropout': 0.0,
        'batch_size': 512,
        'recurrent_dropout': 0.0,
        'epochs': 5,
        'dense_activation': 'sigmoid',
        'optimizer': 'RMSprop'
        }
     , 

     'LSTM_CNN' : 
        {
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
     ,

     'GRU' : 
        {
        'model_name' : 'GRU_model',
        'loss': 'binary_crossentropy',
        'num_nueron' : 100,
        'dropout': 0.2,
        'batch_size': 512,
        'dropout': 0.2,
        'pool_length': 2,
        'epochs': 5,
        'activation': 'relu',
        'dense_activation': 'sigmoid',
        'validation_split' : 0.1,
        'optimizer': 'adam'
        }
     ,

    'CNN' : 
        {
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
    
}

model_object = {
    'LSTM': LSTM_(model_name , embeding_matrix , max_words , params[model_name]),
    'LSTM_CNN': LSTM_CNN_(model_name , embeding_matrix , max_words , params[model_name]),
    'CNN': CNN_(model_name , embeding_matrix , max_words , params[model_name]),
    'GRU': GRU_(model_name , embeding_matrix , max_words , params[model_name])
}

model_ = model_object[model_name]
model_.build_model()
print("----model build----- ")
print("----model summary----")
print(model_.model.summary())
# Train the model ---> save the weights with best validation loss
print("----training.....")
model_.train(X_train, Y_train, epochs=params[model_name]["epochs"], batch_size=params[model_name]["batch_size"])
print("------model trained------")
print("++++++++++++++++++++++++++++++++++")
print(model_name)
print("++++++++++++++++++++++++++++++++++")

print("-------generate labels for test data--------")
model_.load()
print(model_.model.summary())

model_.prediction(X_test, model_name, batch_size=1024)





