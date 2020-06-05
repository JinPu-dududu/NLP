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
from wordcloud import WordCloud, STOPWORDS
from yellowbrick.text import DispersionPlot

### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))


##### wordcloud
def wordcloud_user(texts):
    text = texts.replace("|||","").lower()
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    wordcloud =  WordCloud(stopwords=STOPWORDS,background_color='white',width=3000,height=3000).generate(text)
    plt.imshow(wordcloud)          
    plt.axis('off')           
    plt.show()
    
def wordcloud(user_dict):
    text = ""
    for user,content in user_dict.items():
        text += content
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    wordcloud =  WordCloud(stopwords=STOPWORDS,background_color='white',width=3000,height=3000).generate(text.lower())
    plt.imshow(wordcloud)          
    plt.axis('off')
    plt.show()

user = # find a user
wordcloud_user(tweets[user])

wordcloud(tweets)

