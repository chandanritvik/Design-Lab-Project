#imports 
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer

def preprocessing(tweet):
    tweet = re.sub(r'-',' ',tweet)
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)','', tweet, flags=re.MULTILINE) #to remove links that start with HTTP/HTTPS in the tweet 
    tweet = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,61}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)','', tweet, flags=re.MULTILINE) # to remove other url links
    tweet = re.sub(r'(\d|\d\d):\d\d(am|pm)','',tweet,flags=re.MULTILINE)
    tweet = re.sub(r'(\s|.)#\w+', r'\1', tweet)
    # tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('#'))
    # tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('&'))
    tweet = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",tweet).split()) #to remove #, @ and # 
    tweet = tweet.lower() #to lower the tweets 
    stemmer= PorterStemmer()
    return ' '.join([stemmer.stem(x) for x in tweet.split()])

# preprocessing("#hello hi-there")

# to parse data
def read_data(path, mode = "train"):
    if mode == "test":
        tweetid = []
        tweet = []
        with open(path,'r') as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                t , text = line.split("\t")
                tweetid.append(t)
                tweet.append(text)

        df = pd.DataFrame({"id":tweetid,"text":tweet})
    elif mode =="train" :
        tweetid = []
        tweet = []
        label =[]
        with open(path,'r') as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                t , text , l = line.split("\t")
                tweetid.append(t)
                tweet.append(text)
                label.append(l)

        df = pd.DataFrame({"id":tweetid,"text":tweet,"hateful":label})

    return df

def get_data(data = 'train'):

    if data not in ['train', 'test']:
        raise Exception("Parameter should be 'train' or 'test'")
    traindf = read_data("../Data/train.tsv",mode='train')
    testdf = read_data("../Data/test.tsv",mode='test')

    traindf['processed_tweet'] = traindf.apply(lambda row : preprocessing(row['text']),axis = 1 )
    testdf['processed_tweet'] = testdf.apply(lambda row : preprocessing(row['text']),axis = 1 )

    return traindf, testdf


if __name__ == '__main__':
    df_train, df_test = get_data()
    print('Number of train samples are {0}'.format(len(dftrain)))
    print('Number of test samples are {0}'.format(len(dftest)))