import pandas as pd
import re
import json
import sys
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize , sent_tokenize
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer


def read_data(FULL=False , GLOVE_DIMENSION=25):
    """
    This function read train pos, train neg, test and glove regarding dimension
    Input:
        FULL: Boolean, specify what amount of train files will be read
        GLOVE_DIMENSION: Integer number, specify which dimension of glove will be used
    Output:
        3 DataFrame for train pos, train neg, and test 
        1 dictionary for glove embedding
    """

    # Loading the relative paths
    with open("paths/relative_paths.json") as f:
        PATHS = json.load(f)

    if FULL:
        train_pos_file = pd.read_csv(PATHS["train_pos_full"] , header=None  , engine='python' , sep='k760#7*&^')
        train_neg_file = pd.read_csv(PATHS["train_neg_full"] , header=None  , engine='python' , sep='k760#7*&^')
    else:
        train_pos_file = pd.read_csv(PATHS["train_pos"] , header=None  , engine='python' , sep='k760#7*&^')
        train_neg_file = pd.read_csv(PATHS["train_neg"] , header=None  , engine='python' , sep='k760#7*&^')

    test_file = pd.read_csv(PATHS["test"] , header=None , engine='python' , sep='k760#7*&^')

    glove_embeddings_path = PATHS["glove_folder"] + '/glove.twitter.27B.' + str(GLOVE_DIMENSION) + 'd.txt'        
    glove_embeddings = {}
    with open(glove_embeddings_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            glove_embeddings[word] = coefs

    return train_pos_file, train_neg_file, test_file, glove_embeddings

def preprocessing (data_frame):
    """
    This fuction provides required preprocessing task for a dataframe including tweets and return preprocessed dataframe
    """

    # replace links with NULL.
    data_frame = data_frame.replace(r'http\S+', "", regex=True).replace(r'www\S+', "", regex=True)

    # replace smiling face with 'smile' text. smile face: )dD
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", "smile" , regex=True)

    # replace lol face with 'laugh' text. lol face: pP
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[pP]+", "laugh" , regex=True)
    
    # replace sad face with 'saD' text. sad face: (
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", "sad" , regex=True)
    
    # replace neutral face with 'neutral' text. neutral face: \/|l
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[\/|l]+", "neutral" , regex=True)
    
    # seperate concatenated words wih /. Ex.: sad/happy -> sad happy
    data_frame = data_frame.replace(r"/" , " / " , regex=True)
    
    # replace <3 symbol with 'heart' text.
    data_frame = data_frame.replace(r"<3" , "heart" , regex=True)
    
    # replace numbers with NULL.
    data_frame = data_frame.replace(r"[-+]?[.\d]*[\d]+[:,.\d]*" , "" , regex=True )
    
    # remove repeated punctuation. Ex.: ????? -> ? 
    data_frame = data_frame.replace(r"([!?.]){2,}" , r"\1" , regex=True)
    
    # remove elongated endings. Ex.: goodddd -> good 
    data_frame = data_frame.replace(r"\b(\S*?)(.)\2{2,}\b", r"\1", regex=True)
    
    # remove apostrophes
    data_frame = data_frame.replace(r"'" , "" , regex=True)

    # remove <user>
    data_frame = data_frame.replace(r"<user>" , "" , regex=True)

    # remove <url>
    data_frame = data_frame.replace(r"<url>" , "" , regex=True)

    # replace multiple spaces with one
    data_frame = data_frame.replace("\s+", " " , regex=True)

    # convert to lowercase
    data_frame = data_frame.apply(lambda x: x.astype(str).str.lower())
    """
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    data_frame = data_frame.replace(pat, "" , regex=True )
    
    # stemming
    ps = PorterStemmer() 
    data_frame = data_frame.iloc[:,0].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()] ))

    # lemmatization
    lmtzr = WordNetLemmatizer()
    data_frame = data_frame.apply(lambda x: ' '.join([lmtzr.lemmatize(word,'v') for word in x.split()]))
    """
    # remove punctuations 
    # data_frame = data_frame.replace(r".", "" , regex=True)
    # data_frame = data_frame.replace(r",", "" , regex=True)
    # data_frame = data_frame.replace(r":", "" , regex=True)
    # data_frame = data_frame.replace(r";", "" , regex=True)
    # #data_frame = data_frame.replace(r'?', '' , regex=True)
    # data_frame = data_frame.replace(r"!", "" , regex=True)
    # data_frame = data_frame.replace(r">", "" , regex=True)
    # data_frame = data_frame.replace(r"<", "" , regex=True)


    return data_frame

def generate_data(FULL=False , GLOVE_DIMENSION=25 , MAX_WORDS=40):
    """
    This function correspond each word to a number and generate embedding matrix for them --> saved in "embedding_matrix"
    Concatenate pos and neg train tweets and generate a sequence of numbers for them --> saved in "X_train_seq"
    Create a vector of label correspond to train tweets including 1 for pos and 0 for neg --> saved in "Y_train"
    Generate a sequence of nembers for test tweets --> saved in "X_test_seq" 
    """

    train_pos_file, train_neg_file, test_file, glove_embedings = read_data(FULL , GLOVE_DIMENSION)
    preprocessed_train_pos = preprocessing(train_pos_file)#.values.to_numpy()
    preprocessed_train_neg = preprocessing(train_neg_file)#.values.to_numpy()
    preprocessed_test = preprocessing(test_file)#.values.to_numpy()
    data_list = [preprocessed_train_pos , preprocessed_train_neg]
    train_tweets = pd.concat(data_list, ignore_index=False)
    test_tweets = preprocessed_test

    labels = []
    for i in range(preprocessed_train_pos.shape[0]):
        labels.append(1)
     
    for i in range(preprocessed_train_neg.shape[0]):
        labels.append(0)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_tweets[0])

    word_index_train = tokenizer.word_index
    
    print('Found %s unique tokens in train data set.' % len(word_index_train))
    
    sequences_train = tokenizer.texts_to_sequences(train_tweets[0])

    train_data = pad_sequences(sequences_train, maxlen=MAX_WORDS)
    
    print('Shape of train data set tensor:', train_data.shape)
    
    ### TEST
    
    sequences_test = tokenizer.texts_to_sequences(test_tweets[0])
    test_data = pad_sequences(sequences_test, maxlen=MAX_WORDS)
    print('Shape of test data set tensor:', test_data.shape)
    
    hits = 0
    number_of_vocabularies = len(word_index_train)
    embedding_matrix = np.zeros((number_of_vocabularies + 1, GLOVE_DIMENSION))
    for word, idx in word_index_train.items():
        if word in glove_embedings:
            emb = glove_embedings[word]
            embedding_matrix[idx] = emb[-GLOVE_DIMENSION:]
            hits += 1
        else:
            ran_floats = np.random.rand(GLOVE_DIMENSION) * (13.3-0.5) + 0.5
            emb = ran_floats #glove_embedings["unk"]
            embedding_matrix[idx] = emb

    embedding_matrix[number_of_vocabularies] = [0]*GLOVE_DIMENSION
    print('%s words of %s found' % (hits, number_of_vocabularies))

    print('Saving ...')
    dir_name = "data/generated_gloved_%s_words_%s_full_%s" % (GLOVE_DIMENSION, MAX_WORDS, FULL)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    np.save(os.path.join(dir_name, "embedding_matrix"), embedding_matrix)
    np.save(os.path.join(dir_name, "X_train_seq"), train_data)
    np.save(os.path.join(dir_name, "Y_train"), labels)
    np.save(os.path.join(dir_name, "X_test_seq"), test_data)

    print("Saving done.")

def load_data(FULL=False , GLOVE_DIMENSION=25 , MAX_WORDS=40):
    """
    Loading data
    First check if desired folder exist, only load it, otherwise generate and then load
    """
    
    path = "data/generated_gloved_%s_words_%s_full_%s" % (GLOVE_DIMENSION, MAX_WORDS, FULL)
    if not os.path.isdir(path):
        print("Generating data for new parameters")
        generate_data(FULL,GLOVE_DIMENSION,MAX_WORDS)
        print("Generating data done!")
    else:
        print("Data generated before...")
    embedding_matrix = np.load(os.path.join(path, "embedding_matrix.npy") , allow_pickle=True)
    X_train = np.load(os.path.join(path, "X_train_seq.npy") , allow_pickle=True)
    Y_train = np.load(os.path.join(path, "Y_train.npy") , allow_pickle=True)
    print("y shape: ")
    print(Y_train.shape)
    X_test = np.load(os.path.join(path, "X_test_seq.npy") , allow_pickle=True)
    return X_train, Y_train, X_test, embedding_matrix

