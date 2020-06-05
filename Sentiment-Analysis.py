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


### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))



##### Sentiment Analysis: extract emotional information and compare between different users/time periods
### Naive Sentiment Analysis
def sentiment_words(url):
    request = requests.get(url)
    print("retriving data >>> status code: ",request.status_code)
    text = request.text
    word_list = text[text.find("\n\n")+2:].split("\n")
    return word_list

def sentiment_scores(tokens,pos_list,neg_list):
    cpos, cneg = 0,0
    for token in tokens:
        if token in pos_list:
            cpos += 1
        elif token in neg_list:
            cneg += 1
    return cpos,cneg

def sentiment_comparison_user(user_dic,pos_list,neg_list,n_track=50):
    comparison = []
    i = 0
    start_time = time.time()
    for user,text in user_dic.items():
        i += 1
        if i%n_track==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()
        tokens = nltk.word_tokenize(text.replace("|||",""))
        tokens = [word.lower() for word in tokens]
        cpos,cneg = sentiment_scores(tokens,pos_list,neg_list)
        comparison.append([user,cpos,cneg,len(tokens)])
    comparison = pd.DataFrame(comparison,columns=["user","cpos","cneg","length"]).set_index("user")
    # this function takes some time thus set alarm to inform me when it is done
    duration = 1000
    freq = 440
    winsound.Beep(freq, duration)
    return comparison

def sentiment_comparison_time(texts,pos_list,neg_list,n_track=1000):
    comparison = []
    i = 0
    text = texts.split("|||")
    start_time = time.time()
    for t in text:
        i += 1
        if i%n_track==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()
        tokens = nltk.word_tokenize(t)
        tokens = [word.lower() for word in tokens]
        cpos,cneg = sentiment_scores(tokens,pos_list,neg_list)
        comparison.append([i,cpos,cneg,len(tokens)])
    comparison = pd.DataFrame(comparison,columns=["time","cpos","cneg","length"]).set_index("time")
    return comparison
            
pos_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
neg_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'
pos_list = sentiment_words(pos_url)[:-1]
neg_list = sentiment_words(neg_url)[:-1]


comparison_user_naive = sentiment_comparison_user(tweets,pos_list,neg_list,n_track=50)
comparison_user_naive.to_csv("comparison_user_naive.csv")


## Output graph 1: how users are feeling?
comparison_user_naive["score"] = (comparison_user_naive["cpos"] - comparison_user_naive["cneg"])/np.maximum(comparison_user_naive["length"],1e-9)
comparison_user_naive["ppos"] = comparison_user_naive["cpos"]/comparison_user_naive["length"]
comparison_user_naive["pneg"] = comparison_user_naive["cneg"]/comparison_user_naive["length"]

fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.scatter(comparison_user_naive["ppos"]*100, 
            comparison_user_naive["pneg"]*100, 
            s=comparison_user_naive["length"]/800, 
            c="blue",alpha=0.5, edgecolors="grey", linewidth=2)
ax.set_xlim(0,8)
ax.set_ylim(0,8)
ax.set_xlabel("percentage of postive sentiments (%)",fontsize="medium")
ax.set_ylabel("percentage of negative sentiments (%)",fontsize="medium")
ax.text(6.5,7,"size: tweet length",fontsize="medium")


## Output graph 2: the sentimental change of one user
user = 1159308178875670528
comparison_time_naive = sentiment_comparison_time(tweets[user],pos_list,neg_list)
comparison_time_naive["score"] = (comparison_time_naive["cpos"] - comparison_time_naive["cneg"])/np.maximum(comparison_time_naive["length"],1e-9)
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(comparison_time_naive.index,comparison_time_naive.score)
# ax.text(2500,-0.4,f"user:{user}")
ax.text(2500,-0.45,f"mean:{np.mean(comparison_time_naive['score']):.2%}")
plt.show()


### NRC Analysis
def nrc_emotions(path):
    with open(path,"r") as file:
        i = 0
        while i <46:
            next(file)
            i += 1
        emotions = file.read().split("\n")[:-1]
    emotion_dict = dict()
    emotion_dim = set()
    for e in emotions:
        line = e.split("\t")
        if line[2] == "1":
            emotion_dim = emotion_dim.union(([line[1]]))
            if emotion_dict.get(line[0]):
                emotion_dict[line[0]].append(line[1])
            else:
                emotion_dict[line[0]]=[line[1]]
    emotion_index = dict()
    for i,e in enumerate(emotion_dim):
        emotion_index[e]=i
    return emotion_index,emotion_dict

def emotions_scores(tokens,emotion_index,emotion_dict):
    emotion_scores = np.zeros(len(emotion_index))
    for token in tokens:
        if emotion_dict.get(token):
            for emotion in emotion_dict[token]:
                index = emotion_index[emotion]
                emotion_scores[index] += 1
    return emotion_scores

def emotions_comparison_user(user_dic,emotion_index,emotion_dict,n_track=100):
    comparison = []
    i = 0
    start_time = time.time()
    for user,text in user_dic.items():
        i += 1
        if i%n_track==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()
        tokens = nltk.word_tokenize(text.replace("|||",""))
        tokens = [word.lower() for word in tokens]
        emotion_scores = emotions_scores(tokens,emotion_index,emotion_dict)
        emotion_scores_ = list(emotion_scores)
        emotion_scores_.insert(0,user)
        emotion_scores_.append(len(tokens))
        comparison.append(emotion_scores_)
    columns = ["user"]+list(emotion_index.keys())+["length"]
    comparison = pd.DataFrame(comparison,columns=columns).set_index("user")
    # this function takes some time thus set alarm to inform me when it is done
    duration = 1000
    freq = 440
    winsound.Beep(freq, duration)
    return comparison

def emotion_comparison_time(texts,emotion_index,emotion_dict,n_track=1000):
    comparison = []
    i = 0
    text = texts.split("|||")
    start_time = time.time()
    for t in text:
        i += 1
        if i%n_track==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()
        tokens = nltk.word_tokenize(t)
        tokens = [word.lower() for word in tokens]
        emotion_scores = emotions_scores(tokens,emotion_index,emotion_dict)
        emotion_scores_ = list(emotion_scores)
        emotion_scores_.insert(0,i)
        emotion_scores_.append(len(tokens))
        comparison.append(emotion_scores_)
    columns = ["time"]+list(emotion_index.keys())+["length"]
    comparison = pd.DataFrame(comparison,columns=columns).set_index("time")
    return comparison
                
path = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emotion_index,emotion_dict = nrc_emotions(path)


comparison_user_nrc = emotions_comparison_user(tweets,emotion_index,emotion_dict,n_track=200)
comparison_user_nrc.to_csv("comparison_user_nrc.csv")


## Output graph 1: how users are feeling? (postive and negative)
comparison_user_nrc_norm = comparison_user_nrc[list(emotion_index.keys())].copy()
for i in list(emotion_index.keys()):
    comparison_user_nrc_norm[i] = comparison_user_nrc[i]/comparison_user_nrc["length"]
pos_emotion = ["surprise","anticipation","joy","trust"]
neg_emotion = ["sadness","fear","anger","disgust"]
biemotion_user = pd.DataFrame()
biemotion_user["pos_emotion"]= comparison_user_nrc_norm.apply(lambda x:np.sum(x[pos_emotion]),axis=1)
biemotion_user["neg_emotion"]= comparison_user_nrc_norm.apply(lambda x:np.sum(x[neg_emotion]),axis=1)
biemotion_user["length"]= comparison_user_nrc["length"]
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.scatter(biemotion_user["pos_emotion"]*100, 
            biemotion_user["neg_emotion"]*100, 
            s=biemotion_user["length"]/800, 
            c="blue",alpha=0.5, edgecolors="grey", linewidth=2)
ax.set_xlim(0,25)
ax.set_ylim(0,25)
ax.set_xlabel("percentage of postive sentiments (%)")
ax.set_ylabel("percentage of negative sentiments (%)")
ax.text(21.5,21.5,"size: tweet length")


## Output graph 2: the emotional radar plot of one user/multiple users
def radar_emotion_plot(users,comparison_user_nrc_norm):
    fig = go.Figure()
    for user in users:
        fig.add_trace(go.Scatterpolar(
              r=list(comparison_user_nrc_norm.loc[user,pos_emotion+neg_emotion]),
              theta=pos_emotion+neg_emotion,
              fill='toself',
              name='User 1'))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
        )),
      showlegend=False)
    fig.show()

user = # find a user
user2 = # find a user
user3 = # find a user
radar_emotion_plot([user,user2],comparison_user_nrc_norm)


## Output graph 3: the emotional change one user
comparison_time_nrc = emotion_comparison_time(tweets[user],emotion_index,emotion_dict)
fig,ax = plt.subplots(1,1,figsize=(12,6))
biemotion_time = (np.sum(comparison_time_nrc[pos_emotion],axis=1)-np.sum(comparison_time_nrc[neg_emotion],axis=1))/np.maximum(comparison_time_nrc["length"],1e-9)
biemotion_time.plot(ax=ax)
ax.text(2500,2.5,f"user:{user}")


### VADER
def vader_sentiment(user_dict,n_track=100):
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    i = 0
    start_time = time.time()
    for user,text in user_dict.items():
        i += 1
        if i%n_track ==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()    
        pos=neg=neu=compound=0
        text = text.replace("|||",".")
        sentences = nltk.sent_tokenize(text.lower())
        for sentence in sentences:
            vs = analyzer.polarity_scores(sentence)
            pos += vs["pos"]/len(sentences)
            neg += vs["neg"]/len(sentences)
            neu += vs["neu"]/len(sentences)
            compound += vs["compound"]/len(sentences)
        sentiments.append([user,pos,neg,neu,compound])
    sentiments = pd.DataFrame(sentiments,columns=["user","pos","neg","neu","compound"]).set_index("user")
    # this function takes some time thus set alarm to inform me when it is done
    duration = 1000
    freq = 440
    winsound.Beep(freq, duration)
    return sentiments

def vader_sentiment_time(texts,n_track=1000):
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()
    text = texts.split("|||")
    i = 0
    start_time = time.time()
    for t in text:
        i += 1
        if i%n_track ==0:
            print(f"Progress: {i}th iteration")  # track progress
            print(f">>>>>> time: {time.time() - start_time}")  # record operating time every "n_track" iterations
            start_time = time.time()    
        pos=neg=neu=compound=0
        sentences = nltk.sent_tokenize(t.lower())
        for sentence in sentences:
            vs = analyzer.polarity_scores(sentence)
            pos += vs["pos"]/len(sentences)
            neg += vs["neg"]/len(sentences)
            neu += vs["neu"]/len(sentences)
            compound += vs["compound"]/len(sentences)
        sentiments.append([i,pos,neg,neu,compound])
    sentiments = pd.DataFrame(sentiments,columns=["time","pos","neg","neu","compound"]).set_index("time")
    return sentiments



comparison_user_vader = vader_sentiment(tweets)
comparison_user_vader.to_csv("comparison_user_vader.csv")



## Output graph 1: how users are feeling?
fig, axes = plt.subplots(1,2,figsize=(18,6))
axes[0].scatter(comparison_user_vader["pos"]*100, 
            comparison_user_vader["neg"]*100,
            s=comparison_user_vader["compound"]*100, 
            c="blue",alpha=0.5, edgecolors="grey", linewidth=2)
axes[0].set_xlabel("percentage of postive sentiments (%)",fontsize="medium")
axes[0].set_ylabel("percentage of negative sentiments (%)",fontsize="medium")

axes[1].scatter(comparison_user_vader["pos"]*100, 
            comparison_user_vader["neu"]*100,
            s=comparison_user_vader["compound"]*100, 
            c="blue",alpha=0.5, edgecolors="grey", linewidth=2)
axes[1].set_xlabel("percentage of postive sentiments (%)",fontsize="medium")
axes[1].set_ylabel("percentage of neutral sentiments (%)",fontsize="medium")
plt.show()



## Output graph 2: the emotional radar plot of one user/multiple users
def radar_emotion_plot(users,comparison_user_vader):
    fig = go.Figure()
    for user in users:
        fig.add_trace(go.Scatterpolar(
              r=list(comparison_user_vader.loc[user,:][["pos","neg","neu"]]),
              theta=["positive","negative","neutral"],
              fill='toself',
              name='User 1'))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
        )),
      showlegend=False)
    fig.show()


radar_emotion_plot([user,user2,user3],comparison_user_vader)



## Output graph 3: the emotional radar plot of one user
comparison_time_vader = vader_sentiment_time(tweets[user])
fig,ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(comparison_time_vader.index,comparison_time_vader["compound"])
ax.text(2500,-0.9,f"user:{user}")
ax.text(2500,-1,f"mean:{np.mean(comparison_time_vader['compound']):.2%}")
plt.show()

