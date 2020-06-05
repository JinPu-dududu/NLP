#!/usr/bin/env python
# coding: utf-8


import pickle
import nltk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gensim

### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))


### text summarization
print(gensim.summarization.summarize(tweets[list(tweets.keys())[2]].replace("|||","."),ratio=0.0003))
print(gensim.summarization.keywords(tweets[list(tweets.keys())[2]].replace("|||","."),words=10))

