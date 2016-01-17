import numpy as np
import pandas as pd
import os
from keras.preprocessing import sequence

data_path = '/data3/flickr30k/'
annotation_path = os.path.join(data_path, 'results_20130124.token') # pickle

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
captions = annotations['caption'].values

word2idx = np.load('/data3/flickr30k/word2idx.npy').tolist()
feats = np.load('/data3/flickr30k/feats.npy')

index = np.arange(len(feats))
np.random.shuffle(index)

feats = feats[index]
captions = captions[index]


maxlen = 30
batch_size = 3
len_feat = batch_size * 2


for start, end in zip(range(0,len_feat,batch_size),range(batch_size,len_feat,batch_size)):
    # start, end : 0,3
    current_feats = feats[start:end] # (batch_size,4096)
    current_captions = captions[start:end]

    #current_caption_ind : list type / len(curr~) : 3
    current_caption_ind = map(lambda cap: [word2idx[word] for word in cap.lower().split(' ')[:-1] if word in word2idx], current_captions)


    # current_caption_matrix : ndarray / shape : (3,31)
    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1) # ndarray

    # current_caption_matrix : ndarray / shape : (3,32) / front => 0
    current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)

    # current_caption_matrix.shape[0] : batch_size
    # current_caption_matrix.shape[1] : maxlen = maxlen + 1
    current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))

    # nonzeros : ndarray / (3,)
    nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_caption_matrix ))


    for ind, row in enumerate(current_mask_matrix):
        print (ind,row)
        print nonzeros[ind]
        row[:nonzeros[ind]] = 1
        print row
        print type(row)