import pandas as pd
import re
import json

def read_data(FULL=False , GLOVE_DIMENSION=25):

    # Loading the relative paths
    with open("config/relative_paths.json") as f:
        PATHS = json.load(f)

    if FULL:
        train_pos_file = pd.read_csv(PATHS["train_pos_full"])
        train_neg_file = pd.read_csv(PATHS["train_neg_full"])
    else:
        train_pos_file = pd.read_csv(PATHS["train_pos"] , header=None  , engine='python' , sep='k760#7*&^')
        train_neg_file = pd.read_csv(PATHS["train_neg"] , header=None  , engine='python' , sep='k760#7*&^')

    test_file = pd.read_csv(PATHS["test"] , header=None  , engine='python' , sep='k760#7*&^')

    # reading glove embedings from directory
    glove_embeddings_path = PATHS["glove_folder"] + '/glove.twitter.27B.' + str(GLOVE_DIMENSION) + 'd.txt'
    glove_embeddings_file = pd.read_csv(glove_embeddings_path , header=None , sep = '\s+')
    glove_array = glove_embeddings_file.to_numpy()
    
    glove_embeddings = {glove_array[i][0]: glove_array[i][1:] for i in range(glove_array.shape[0])}
    #print(glove_embeddings)
    
    

    #print({glove_embeddings_file[i][0]: glove_embeddings_file[i][1:].tolist() for i in glove_embeddings_file.T})
    #print(glove_embeddings)
    #print(glove_embeddings_file['work'])

    return train_pos_file, train_neg_file, test_file, glove_embeddings

def preprocessing (data_frame):

    # replace links with '<url>' text.
    data_frame = data_frame.replace(r'http\S+', "<url>", regex=True).replace(r'www\S+', "<url>", regex=True)

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
    data_frame = data_frame.replace(r"<3" , "<heart>" , regex=True)
    
    # replace numbers with '<number>' text.
    data_frame = data_frame.replace(r"[-+]?[.\d]*[\d]+[:,.\d]*" , "<number>" , regex=True )
    
    # replace repeated punctuation with '<repeat>' text. Ex.: ????? -> ? <repeat>
    data_frame = data_frame.replace(r"([!?.]){2,}" , r"\1 <repeat>" , regex=True)
    
    # replace elongated endings with '<elong>' text. Ex.: goodddd -> good <elong>
    data_frame = data_frame.replace(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ", regex=True)
    
    # replace multiple spaces with one
    data_frame = data_frame.replace("\s+", " " , regex=True)
    
    # remove apostrophes
    data_frame = data_frame.replace(r"'" , "" , regex=True)

    return data_frame


def load_data(FULL=False , GLOVE_DIMENSION=25):

    train_pos_file, train_neg_file, test_file, glove_embedings = read_data(FULL , GLOVE_DIMENSION)
    preprocessed_train_pos = preprocessing(train_pos_file)
    preprocessed_train_neg = preprocessing(train_neg_file)
    preprocessed_test = preprocessing(test_file)

    # df = pd.read_csv("test.txt")
    # df = preprocessing(df)
    # print(df)
    return preprocessed_train_pos, preprocessed_train_neg, preprocessed_test, glove_embedings

#read_data(GLOVE_DIMENSION=1)





