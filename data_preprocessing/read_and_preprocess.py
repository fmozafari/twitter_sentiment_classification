import pandas as pd
import re
import json

def read_data(FULL=False):

    # Loading the relative paths
    # with open("config/relative_paths.json") as f:
    #     PATHS = json.load(f)

    # if FULL:
    #     train_pos_file = pd.read_csv(PATHS["train_pos_full"])
    #     train_neg_file = pd.read_csv(PATHS["train_neg_full"])
    # else:
    #     train_pos_file = pd.read_csv(PATHS["train_pos"])
    #     train_neg_file = pd.read_csv(PATHS["train_neg"])

    # test_file = pd.read_csv(PATHS["test"])

    #preprocessed_train_pos = preprocessing(train_pos_file)
    #preprocessed_train_neg = preprocessing(train_neg_file)
    #preprocessed_test = preprocessing(test_file)

    df = pd.read_csv("test.txt")
    df = preprocessing(df)
    print(df)

def preprocessing (data_frame):

    # replace links with '<url>' text.
    data_frame = data_frame.replace(r'http\S+', "<url>", regex=True).replace(r'www\S+', 'url', regex=True)

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


read_data()




