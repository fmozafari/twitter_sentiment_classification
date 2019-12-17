import sys
sys.path.insert(1,'models/')
from base_model import base_model
from keras.models import Sequential
from keras.layers import *

class LSTM_CNN_(base_model):

    def __init__(self, name , embeding_matrix , tweet_len , params):
        """
        Call base_model constructor
        """
        super().__init__( name , embeding_matrix , tweet_len , params)

    def build_model(self):
        """
        Build model for combined LSTM and CNN
        """
        # Model type
        self.model = Sequential()
        # Add embedding layer
        self.model.add(Embedding(self.embeding_matrix.shape[0], self.embeding_matrix.shape[1], input_length=self.tweet_len, 
                                 weights=[self.embeding_matrix], name='emb'))
        # Add LSTM layer
        self.model.add(LSTM(units=self.params["LSTM_num_neurons"], dropout=self.params["LSTM_dropout"], recurrent_dropout=self.params["LSTM_recurrent_dropout"], return_sequences=True))
        # Add one dimensioanl convolution layer 
        self.model.add(Conv1D(filters=self.params["CNN_filters"], kernel_size=self.params["CNN_kernel_size"], activation=self.params["CNN_activation"]))
        # Add one dimensional max pooling layer
        self.model.add(MaxPooling1D(pool_size=self.params["CNN_pool_size"]))
        # Add flatten layer
        self.model.add(Flatten())
        # Add dense layer to predict output
        self.model.add(Dense(1, activation=self.params["DENSE_activation"]))
        # Compile the model
        self.model.compile(loss=self.params["loss"], optimizer=self.params["optimizer"], metrics=['accuracy'])