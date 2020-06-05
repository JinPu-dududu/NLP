#!/usr/bin/env python
# coding: utf-8

import pickle
import nltk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.stem.porter import *
p_stemmer = PorterStemmer()
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim import corpora, models, similarities
from gensim.models.ldamodel import LdaModel
import pprint
import pyLDAvis.gensim
from gensim.similarities.docsim import Similarity



### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))


##### topic analysis
words_list = []
users = []
for user,text in tweets.items():
    users.append(user)
    words = nltk.word_tokenize(text.replace("|||","").lower())
    words = [word for word in words if word not in STOPWORDS and word.isalnum() and len(word)>=2]
    words_list.append(words)


# # use only one user's tweets
# words_list = []
# for i in tweets[list(tweets.keys())[2]].split("|||"):
#     words =[word for word in nltk.word_tokenize(i) if word not in STOPWORDS and word.isalnum() and len(word)>=2]
#     words_list.append(words)


num_topics = 3
dictionary = corpora.Dictionary(words_list)
corpus = [dictionary.doc2bow(words) for words in words_list]
lda = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)



###output1: topics and corresponding words
pp = pprint.PrettyPrinter(indent=4)    
pp.pprint(lda.print_topics(num_words=10))



###output2: 2 ways of showing one topic and corresponding words
lda.print_topic(topicno=0)    
lda.show_topic(1)


### ouput3: show topic of one user (even new user)
sorted(lda.get_document_topics(corpus[100],minimum_probability=0,per_word_topics=False),key=lambda x:x[1],reverse=True) 


### output4: visualize LDA
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, R=15, sort_topics=False)     
pyLDAvis.display(lda_display)


##### Text Similarities
doc = tweets[list(tweets.keys())[2]].replace("|||","")
lsi = models.LsiModel(corpus,id2word=dictionary, num_topics=3)  
words_new = nltk.word_tokenize(doc.lower())
words_new = [word for word in words if word not in STOPWORDS and word.isalnum() and len(word)>=2]
vec_bow = dictionary.doc2bow(words_new)     
vec_lsi = lsi[vec_bow]
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]       
sims = sorted(enumerate(sims), key=lambda item: -item[1])
sims

