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
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



### read files
tweets = pickle.load(open( "tweet_2292.pickle", "rb" ))

print(len(tweets.keys()))
print(list(tweets.keys())[0])
tweets[list(tweets.keys())[0]]



### select a user
sample = tweets[list(tweets.keys())[774]]



### data cleaning
sentences = [sentence for sentence in sample.split("|||") if len(nltk.word_tokenize(sentence))>=1]
for i in range(len(sentences)):
    sentence = sentences[i].strip().lower()
    words = sentence.split(" ")
    words = [word for word in words if word not in STOPWORDS and word.isalnum()]
    sentences[i] = " ".join(words)



### find the cutoff point for truncating and cleaning
length = [len(nltk.word_tokenize(lst)) for lst in sentences]
plt.hist(length,bins=10)
plt.show()



sentences = [sentence for sentence in sentences if len(nltk.word_tokenize(sentence))>=10]


embedding_dim = 20
max_length = 15
trunct_type = "post"
padding_type = "pre"
oov_tok = "<OOV>"


### tokenize
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
index_word = {index:word for word,index in word_index.items()}
print("length of word index:",len(word_index))
sequences = tokenizer.texts_to_sequences(sentences)
length2 = [len(lst) for lst in sequences]
print("max number of words:",max(length2))
print("min number of words:",min(length2))


### padding
padded = pad_sequences(sequences,padding=padding_type,truncating=trunct_type,maxlen=max_length)


vocab_size = len(word_index)+1
X = padded[:,:-1]
Y = padded[:,-1]
Y = tf.keras.utils.to_categorical(Y,num_classes=vocab_size)
X.shape,Y.shape



### model building
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(vocab_size,activation="softmax")
])

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=["accuracy"])
model.summary()


model.fit(X,Y,epochs=30,verbose=1)


### predict
choice = random.randint(0,len(padded)) 
seed = padded[choice,1:].reshape(1,max_length-1)
tweet_robot = sentences[choice]
for i in range(5):
    predicted = model.predict_classes(seed,verbose=0)
    seed = np.append(seed,[int(predicted)])[1:].reshape(1,max_length-1)
    tweet_robot = tweet_robot + " " + str(index_word[int(predicted)])

