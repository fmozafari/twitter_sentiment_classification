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


class CNN_():
    def __init__(self, name , embeding_matrix , tweet_len):
        with open("paths/relative_paths.json") as f:
            self.paths = json.load(f)
        self.model_name = name
        self.embeding_matrix = embeding_matrix
        self.tweet_len = tweet_len
        self.model_save_dir = os.path.join(self.paths["model_parameters"], self.model_name)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        # Initialize paths
        self.checkpoint_path = os.path.join(self.model_save_dir, "best_checkpoint.hdf5")
        self.model_json_path = os.path.join(self.model_save_dir, "model.json")

    def build_model(self, params):
        # model type
        self.model = Sequential()
        
        # Add embedding layer for first layer
        self.model.add(Embedding(self.embeding_matrix.shape[0], self.embeding_matrix.shape[1], input_length=self.tweet_len,
                                 weights=[self.embeding_matrix], name='emb'))

        self.model.add(Conv1D(filters=params["filters"] , kernel_regularizer=regularizers.l2(0.01), 
                                kernel_size=params["kernel_size"], activation=params["activation"]))

        self.model.add(MaxPooling1D(pool_size=params["MP_pool_size"]))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(1, activation=params["dense_activation"]))
        
        self.model.compile(loss=params["loss"] , metrics=['accuracy'] , optimizer='adam')

    def train(self, X, Y, epochs, batch_size, validation_data=None):
        # Reproducibility
        from numpy.random import seed
        seed(1)
        import tensorflow
        tensorflow.random.set_seed(2)

        # Save the architecture of model
        model_json = self.model.to_json()
        json.dump(model_json, open(self.model_json_path, "w"))

        # Callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=True, save_weights_only=False)]

        # Train model
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)
        
    def load(self):
        model_json = json.load(open(self.model_json_path, "r"))
        self.model = model_from_json(model_json)
        self.model.load_weights(self.checkpoint_path)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def evaluate_validation(self , X , Y , batch_size):
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def prediction(self, X_test, file_name , batch_size=1024):
        result = pd.DataFrame()
        predicted_classes = self.model.predict_classes(X_test, batch_size=batch_size)
        # Id set
        result["Id"]= range(1, predicted_classes.shape[0]+1)
        # Prediction value
        result["Prediction"] = predicted_classes
        # Replacing 0 value with -1 
        result["Prediction"].replace(0, -1, inplace=True)
        # Save result in results_folder
        result.to_csv(os.path.join(self.paths["results_folder"], file_name + ".csv"))

    

    