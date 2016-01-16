import numpy as np
import pandas as pd
import os


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

    for k in range(len(voca_[i])-1):
        #word_[sent] = voca_[i][k]
        print voca_[i][k]
        vocabulary[voca_[i][k]] = vocabulary.get(voca_[i][k],0) + 1
        sent += 1

caption_data = {}
caption_flipped = {}

for p in range(1,len(vocabulary)):
    if vocabulary.values()[p]>1:
        caption_data[p-1] = vocabulary.keys()[p]



caption_flipped = dict(zip(caption_data.values(),caption_data.keys()))
print caption_flipped
print caption_flipped['.']

