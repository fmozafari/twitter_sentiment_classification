from data_preprocessing.read_and_preprocess import *

# load data
print("Loading Data...")
train_pos, train_neg, test, glove_embedings = load_data()
print("Data Loaded...")