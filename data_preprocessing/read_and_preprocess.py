import pandas as pd
import re
import json
import sys
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

def read_data(FULL=False , GLOVE_DIMENSION=25):

    # Loading the relative paths
    with open("config/relative_paths.json") as f:
        PATHS = json.load(f)

    if FULL:
        train_pos_file = pd.read_csv(PATHS["train_pos_full"] , header=None  , engine='python' , sep='k760#7*&^')
        train_neg_file = pd.read_csv(PATHS["train_neg_full"] , header=None  , engine='python' , sep='k760#7*&^')
    else:
        train_pos_file = pd.read_csv(PATHS["train_pos"] , header=None  , engine='python' , sep='k760#7*&^')
        train_neg_file = pd.read_csv(PATHS["train_neg"] , header=None  , engine='python' , sep='k760#7*&^')

    test_file = pd.read_csv(PATHS["test"] , header=None , engine='python' , sep='k760#7*&^')

    # reading glove embedings from directory
    glove_embeddings_path = PATHS["glove_folder"] + '/glove.twitter.27B.' + str(GLOVE_DIMENSION) + 'd.txt'
    glove_embeddings_file = pd.read_csv(glove_embeddings_path , header=None , sep = '\s+')
    glove_array = glove_embeddings_file.to_numpy()
    
    glove_embeddings = {glove_array[i][0]: glove_array[i][1:] for i in range(glove_array.shape[0])}
    #print(glove_embeddings)

    return train_pos_file, train_neg_file, test_file, glove_embeddings

def preprocessing (data_frame):

    # replace links with '<url>' text.
    data_frame = data_frame.replace(r'http\S+', "", regex=True).replace(r'www\S+', "", regex=True)

    # replace smiling face with '<smile>' text. smile face: )dD
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", "<smile>" , regex=True)

    # replace lol face with '<lolface>' text. lol face: pP
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[pP]+", "<lolface>" , regex=True)
    
    # replace sad face with '<sadface>' text. sad face: (
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", "<sadface>" , regex=True)
    
    # replace neutral face with '<neutralface>' text. neutral face: \/|l
    data_frame = data_frame.replace(r"[8:=;]['`\-]?[\/|l]+", "<neutralface>" , regex=True)
    
    # seperate concatenated words wih /. Ex.: sad/happy -> sad happy
    data_frame = data_frame.replace(r"/" , " / " , regex=True)
    
    # replace <3 symbol with '<heart>' text.
    data_frame = data_frame.replace(r"<3" , "" , regex=True)
    
    # replace numbers with '<number>' text.
    data_frame = data_frame.replace(r"[-+]?[.\d]*[\d]+[:,.\d]*" , "" , regex=True )
    
    # replace repeated punctuation with '<repeat>' text. Ex.: ????? -> ? <repeat>
    data_frame = data_frame.replace(r"([!?.]){2,}" , r"\1" , regex=True)
    
    # replace elongated endings with '<elong>' text. Ex.: goodddd -> good <elong>
    data_frame = data_frame.replace(r"\b(\S*?)(.)\2{2,}\b", r"\1", regex=True)
    
    # replace multiple spaces with one
    data_frame = data_frame.replace("\s+", " " , regex=True)
    
    # remove apostrophes
    data_frame = data_frame.replace(r"'" , "" , regex=True)

    # remove <user>
    data_frame = data_frame.replace(r"<user>" , "" , regex=True)

    # remove <url>
    data_frame = data_frame.replace(r"<url>" , "" , regex=True)

    return data_frame

def load_data(FULL=False , GLOVE_DIMENSION=25 , MAX_WORDS=40):

    train_pos_file, train_neg_file, test_file, glove_embedings = read_data(FULL , GLOVE_DIMENSION)
    preprocessed_train_pos = preprocessing(train_pos_file).to_numpy()
    preprocessed_train_neg = preprocessing(train_neg_file).to_numpy()
    preprocessed_test = preprocessing(test_file).to_numpy()

    labels = []
    train_tweets = []
    for i in range(preprocessed_train_pos.shape[0]):
        labels.append(1)
        train_tweets.append(preprocessed_train_pos[i])
    for i in range(preprocessed_train_neg.shape[0]):
        labels.append(0)
        train_tweets.append(preprocessed_train_neg[i])

    # Mapping every unique word to a integer (bulding the vocabulary)
    print('Bulding the vocabulary...')
    word_to_index = {}
    word_freqs = {}
    k = 0
    for i, tweet in enumerate(train_tweets):
        words = tweet[0].split()
        #print(words)
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                word_to_index[word] = k
                k += 1
            if word not in word_freqs:
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
    word_to_index["unk"] = k
    number_of_vocabularies = len(word_to_index)

    # Converting training tweets to integer sequences...
    train_sequences = []
    for i, tweet in enumerate(train_tweets):
        words = tweet[0].split()
        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])
        train_sequences.append(tweet_seq)

    # Padding the sequences to match the `MAX_WORDS`
    x_train = pad_sequences(train_sequences, maxlen=MAX_WORDS, padding="post", value=number_of_vocabularies)

    # Converting testing tweets to integer sequences...
    test_sequences = []
    for i, tweet in enumerate(preprocessed_test):
        words = tweet[0].split()
        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])
        test_sequences.append(tweet_seq)
    
    # Padding the sequences to match the `MAX_WORDS`
    x_test = pad_sequences(test_sequences, maxlen=MAX_WORDS, padding="post", value=number_of_vocabularies)

    # Generating the embedding matrix for vocabularies 
    unknown = []
    hits = 0
    embedding_matrix = np.zeros((number_of_vocabularies + 1, GLOVE_DIMENSION))
    for word, idx in word_to_index.items():
        if word in glove_embedings:
            emb = glove_embedings[word]
            embedding_matrix[idx] = emb
            hits += 1
        else:
            unknown.append(word)
            ran_floats = np.random.rand(GLOVE_DIMENSION) * (13.3-0.5) + 0.5
            emb = ran_floats#glove_embedings["unk"]
            embedding_matrix[idx] = emb

    embedding_matrix[number_of_vocabularies] = [0]*GLOVE_DIMENSION
    print('%s words of %s found' % (hits, number_of_vocabularies))
    print(embedding_matrix.shape)

    print('Saving everything...')
    dir_name = "data/glove_%s_words_%s_full_%s" % (GLOVE_DIMENSION, MAX_WORDS, FULL)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    np.save(os.path.join(dir_name, "embedding_matrix"), embedding_matrix)
    np.save(os.path.join(dir_name, "X_train"), x_train)
    np.save(os.path.join(dir_name, "Y_train"), labels)
    np.save(os.path.join(dir_name, "X_test"), x_test)

    print("Saving done.")


load_data(FULL=True)





