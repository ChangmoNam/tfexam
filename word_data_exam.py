import numpy as np
import pandas as pd
import os

# NeuralTalk

data_path = '/data3/flickr30k/'
img_path = os.path.join(data_path, 'feats.npy')
annotation_path = os.path.join(data_path, 'results_20130124.token') # pickle

sentence = pd.read_table(annotation_path,sep='\t',header=None,names=['img','voca'])
# sentence : pandas DataFrame
# sentence['voca'] : pandas Series
# sentence['voca'].values
# sentence['voca'].values : np.ndarray

voca_ = {} # dict type
word = sentence['voca'].values # np.ndarray
# word[0] : string type
word_ = {}
vocabulary = {}
sent = 0
for i in range(word.shape[0]):
    voca_[i] = word[i].lower().split(' ') # dict type
    for k in range(len(voca_[i])):
        vocabulary[voca_[i][k]] = vocabulary.get(voca_[i][k],0) + 1
        sent += 1

vocabulary['.']=0
caption_idx2word = {}
caption_word2idx = {}
caption_idx2word[0] = '.'

for p in range(1,len(vocabulary)):
    if vocabulary.values()[p]>1:
        caption_idx2word[p] = vocabulary.keys()[p]

caption_word2idx = dict(zip(caption_idx2word.values(),caption_idx2word.keys()))

np.save('/data3/flickr30k/caption',caption_idx2word)