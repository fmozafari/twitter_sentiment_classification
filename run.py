
from models.LSTM import LSTM_
from models.CNN import CNN_
from data_preprocessing.read_and_preprocess import *

full = True
glove_dimension = 25
max_words = 40

# load data
print("Loading Data...")
X_train, Y_train, X_test, embeding_matrix = load_data(FULL=full , GLOVE_DIMENSION=glove_dimension , MAX_WORDS=max_words)
print("Data Loaded...")

# model_name = "LSTM"
# lstm = LSTM_(model_name , embeding_matrix , max_words)

# lstm.load()
# print(lstm.model.summary())
# xtest = X_test
# lstm.prediction(xtest, model_name, batch_size=1024)

model_name = "CNN"
cnn = CNN_(model_name , embeding_matrix , max_words)

cnn.load()
print(cnn.model.summary())
xtest = X_test
cnn.prediction(xtest, model_name, batch_size=1024)




