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


class CNN_(base_model):
    def __init__(self, name , embeding_matrix , tweet_len , params):
        """
        Call base_model constructor
        """
        super().__init__( name , embeding_matrix , tweet_len , params)

    def build_model(self):
        """
        Build model for CNN
        """
        # model type
        self.model = Sequential()
        
        # Add embedding layer for first layer
        self.model.add(Embedding(self.embeding_matrix.shape[0], self.embeding_matrix.shape[1], input_length=self.tweet_len,
                                 weights=[self.embeding_matrix], name='emb'))
        # Add one dimensional convolution layer
        self.model.add(Conv1D(filters=self.params["filters"] , kernel_regularizer=regularizers.l2(0.01), 
                                kernel_size=self.params["kernel_size"], activation=self.params["activation"]))
        # Add one dimensional max pooling layer
        self.model.add(MaxPooling1D(pool_size=self.params["MP_pool_size"]))
        # Add flatten layer
        self.model.add(Flatten())
        # Add dense layer to predict label
        self.model.add(Dense(1, activation=self.params["dense_activation"]))
        # Compile
        self.model.compile(loss=self.params["loss"] , metrics=['accuracy'] , optimizer='adam')

    