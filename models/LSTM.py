import keras
from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import pandas as pd
import json
import os
from keras import regularizers
import sys
sys.path.insert(1,"models/")
from base_model import base_model


class LSTM_(base_model):
    def __init__(self, name , embeding_matrix , tweet_len , params):
        """
        Call base_model constructor
        """
        super().__init__( name , embeding_matrix , tweet_len , params)

    def build_model(self):
        # Model type
        self.model = Sequential()
        # Add embedding layer
        self.model.add(Embedding(self.embeding_matrix.shape[0] , self.embeding_matrix.shape[1] , 
        input_length=self.tweet_len , weights=[self.embeding_matrix] , name='emb' , trainable=True))
        # Masking layer for pre-trained embeddings
        self.model.add(Masking(mask_value=0.0))
        # Recurrent LSTM layer
        self.model.add(LSTM(units=self.params["num_neurons"], 
                    dropout=self.params["dropout"], recurrent_dropout=self.params["recurrent_dropout"]))

        # Fully connected layer
        #self.model.add(Dense(64, activation='relu'))

        # Dropout for regularization
        self.model.add(Dropout(0.5))
        # Output layer
        self.model.add(Dense(1, activation=self.params["dense_activation"]))
        # Compile the model
        self.model.compile(optimizer=self.params["optimizer"], metrics=['accuracy'] , loss=self.params["loss"])


    

    