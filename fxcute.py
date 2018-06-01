import numpy as np
import sys

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

print('\n\n\n\n\n\n\n\n\n\n\n')

print('#########################################################\n#                                                       #\n#       Demonstration : Morphological Synthesizer       #\n#                                                       #\n#########################################################')

import pyfasttext
from pyfasttext import FastText

print('\nImporting dictionaries...')

dict_drama_token = FastText('vectors/dict_drama_token.bin')

import nltk
from konlpy.tag import Twitter
pos_tagger = Twitter()

def twit_token(doc):
    x = [t[0] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

def twit_pos(doc):
    x = [t[1] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from keras.callbacks import ModelCheckpoint

from keras.layers.normalization import BatchNormalization

from keras.models import load_model

####### Sentence correction
####### drama_mor to drama_raw

def featurize_space(sent,maxlen):
    onehot = np.zeros(maxlen)
    countchar = -1
    for i in range(len(sent)-1):
      if sent[i]!=' ' and i<maxlen:
        countchar=countchar+1
        if sent[i+1]==' ':
          onehot[countchar] = 1
    return onehot

def featurize_corpus(corpus,dic,wdim,maxlen):
    onehot = np.zeros((len(corpus),maxlen))
    conv = np.zeros((len(corpus),maxlen,wdim,1))
    for i in range(len(corpus)):
      if i%1000 == 0:
        print(i)
      onehot[i,:] = featurize_space(corpus[i],maxlen)
      for j in range(len(corpus[i])):
        if j<maxlen and corpus[i][j]!=' ':
          conv[i][j,:,0]=dic[corpus[i][j]]
    return onehot, conv

print('Loading models...')

model_corr = load_model('modelcor/convdnn-05-0.5259.hdf5')

print('\nEnter "bye" to quit\n')

def pred_correction(sent_mor,model,dic,maxlen,wdim):
    onehot = np.zeros((1,maxlen))
    onehot[0] = featurize_space(sent_mor,maxlen)
    conv = np.zeros((1,maxlen,wdim,1))
    for j in range(len(sent_mor)):
      if j<maxlen and sent_mor[j]!=' ':
        conv[0][j,:,0]=dic[sent_mor[j]]
    z = model.predict([onehot,conv])[0]
    sent_raw = ''
    count_char=-1
    for j in range(len(sent_mor)):
      if sent_mor[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent_mor[j]
        if z[count_char]>0.5:
          sent_raw = sent_raw+' '
    return sent_raw

def correct(s):
    x = pred_correction(s,model_corr,dict_drama_token,50,100)+"\n"
    print('>> Output:',x)

while 1:
  s = input('>> Input : ')
  if s == 'bye':
    sys.exit()
  else:
    correct(s)
