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


class base_model():
    """
    Define a base model class for neural network models, 
    then other models will be inherited from this class and modify build_model function 
    """
    def __init__(self, name , embeding_matrix , tweet_len , params=None):
        """
        This constructor initialize model name, embedding matrix for embedding layer, 
        lengh of tweets and required parameters for desired model
        """
        with open("paths/relative_paths.json") as f:
            self.paths = json.load(f)
        self.model_name = name
        self.embeding_matrix = embeding_matrix
        self.tweet_len = tweet_len
        self.params = params
        self.model_save_dir = os.path.join(self.paths["model_parameters"], self.model_name)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        # Initialize paths
        self.checkpoint_path = os.path.join(self.model_save_dir, "best_checkpoint.hdf5")
        self.model_json_path = os.path.join(self.model_save_dir, "model.json")
        self.tensorboard_path = os.path.join(self.model_save_dir, "tensorboard")
        self.history_path = os.path.join(self.model_save_dir, "history")

    def build_model(self):
        """
        Building model that will be modified by desired model
        """
        raise NotImplementedError

    def train(self, X, Y, epochs, batch_size):
        """
        Training procedure for neural networks
        Desired checkpoints nad weights will be saved in a specified path to use for prediction pf test data 
        """
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

        # Create checkpoint callback object
        checkpointer = ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=False, save_weights_only=True)

        # Create tensorboard callback object
        tensorboard = TensorBoard(log_dir=self.tensorboard_path, batch_size=batch_size)

        # Create history callback object
        hist = history_loss()
        hist.set_history_path(self.history_path)

        # Train model
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1,
    shuffle=True, callbacks=[checkpointer, tensorboard, hist])
        
        
    def load(self):
        """
        Load saved model from the desired path
        """
        model_json = json.load(open(self.model_json_path, "r"))
        self.model = model_from_json(model_json)
        self.model.load_weights(self.checkpoint_path)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def evaluate_validation(self , X , Y , batch_size):
        """
        Evaluation of train data
        Return the accuracy of the model on the evaluation data.
        """
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def prediction(self, X_test, file_name , batch_size=1024):
        """
        Generating prediction for test dataset and save it in results folder
        """
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

class history_loss(keras.callbacks.Callback):
    """
    This is a helper callback class.
    Save the accuracy/loss of the train and validation data after every epoch.
    """
    def set_history_path(self, path):
        self.history_path = path
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        if os.path.isfile(self.history_path + ".npy"):
            h1 = np.load(self.history_path + ".npy" , allow_pickle= True ).item()
            merged_history = {}
            for key in h1.keys():
                merged_history[key] = h1[key] + [logs[key]]
            np.save(self.history_path, merged_history)
        else:
            merged_history = {}
            for key in logs.keys():
                merged_history[key] = [logs[key]]
            np.save(self.history_path, merged_history)

    

    
