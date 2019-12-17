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


class GRU_(base_model):
    def __init__(self, name , embeding_matrix , tweet_len , params):
        """
        Call base_model constructor
        """
        super().__init__( name , embeding_matrix , tweet_len , params)

    def build_model(self):
        """
        Build model for GRU
        """
        # Model type
        self.model = Sequential()
        # Add embedding layer
        self.model.add(Embedding(self.embeding_matrix.shape[0], self.embeding_matrix.shape[1], weights=[self.embeding_matrix], 
        input_length=self.tweet_len , trainable=True))
        # GRU layer
        self.model.add(GRU(self.params['num_nueron']))
        # Dropout for regularization
        self.model.add(Dropout(self.params['dropout']))
        # Output layer
        self.model.add(Dense(1,activation=self.params['dense_activation']))
        # Compile the model
        self.model.compile(loss = self.params['loss'], optimizer=self.params['optimizer'], metrics = ['acc'])

        

