#!/usr/bin/env python
# coding: utf-8


import pickle
import nltk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import requests
import winsound
import time
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import sentiwordnet as swn
from nltk.stem.porter import *
p_stemmer = PorterStemmer()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from yellowbrick.text import DispersionPlot
import gensim


### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))


### named entity recognition
def entity_identity(texts,n_track=10000):
    text = texts.replace("|||",".")
    sentences = nltk.sent_tokenize(text)
    entity = []
    i = 0
    start_time = time.time()
    for sentence in sentences:
        i += 1
        if i%n_track==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        chunked = nltk.ne_chunk(tagged)
        for tree in chunked:
            if hasattr(tree,"label"):
                entity.append([tree.label()," ".join(c[0] for c in tree.leaves())])
    entity = pd.DataFrame(entity,columns=['label','entity'])
    # this function takes some time thus set alarm to inform me when it is done
    duration = 1000
    freq = 440
    winsound.Beep(freq, duration)
    return entity

def wordcloud_entity(entity,label="PERSON"):
    text = " ".join(list(entity[entity["label"]==label]["entity"]))
    wordcloud =  WordCloud(stopwords=STOPWORDS,background_color='white',width=3000,height=3000).generate(text)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    plt.imshow(wordcloud)          
    plt.axis('off')
    plt.show()
    
def sentiment_entity(texts,entity="Trump"):
    text = texts.replace("|||",".")
    sentences = nltk.sent_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    pos=neg=neu=compound=count=0
    for sentence in sentences:
        if entity.lower() in sentence.lower():
            vs = analyzer.polarity_scores(sentence)
            pos += vs["pos"]
            neg += vs["neg"]
            neu += vs["neu"]
            compound += vs["compound"]
            count += 1
    return pos/count,neg/count,neu/count,compound/count


texts = "".join([tweet for user,tweet in tweets.items()])
entity = entity_identity(texts)
entity.to_csv("entity.csv")


### output1: wordcloud
wordcloud_entity(entity,label="GPE")

wordcloud_entity(entity,label="PERSON")


### output2: sentiment analysis
pos,neg,neu,compound = sentiment_entity(texts,entity="Trump")
pos2,neg2,neu2,compound2 = sentiment_entity(texts,entity="Polina Shinkina")


print("\t\tpositive\tnegative\tnetrual\t\tcompound")
print(f"Trump\t\t{pos:.2%}\t\t{neg:.2%}\t\t{neu:.2%}\t\t{compound:.2%}")
print(f"Polina Shinkina\t{pos2:.2%}\t\t{neg2:.2%}\t\t{neu2:.2%}\t\t{compound2:.2%}")


pos3,neg3,neu3,compound3 = sentiment_entity(texts,entity="Hong Kong")
pos4,neg4,neu4,compound4 = sentiment_entity(texts,entity="America")
pos5,neg5,neu5,compound5 = sentiment_entity(texts,entity="New York")


print("\t\tpositive\tnegative\tnetrual\t\tcompound")
print(f"America\t\t{pos4:.2%}\t\t{neg4:.2%}\t\t{neu4:.2%}\t\t{compound4:.2%}")
print(f"New York\t{pos5:.2%}\t\t{neg5:.2%}\t\t{neu5:.2%}\t\t{compound5:.2%}")
print(f"Hong Kong\t{pos3:.2%}\t\t{neg3:.2%}\t\t{neu3:.2%}\t\t{compound3:.2%}")

