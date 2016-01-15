import os
#import tensorflow as tf
#import time
import numpy as np
import pandas as pd

data_path = '/data3/flickr30k/'
feat_path = data_path + 'feats.npy'
annotation_path = os.path.join(data_path, 'results_20130124.token')
# word vector

# using Pandas
def get_caption_data(annotation_path, feat_path):
     feats = np.load(feat_path)
     # Pandas read_table ==> read data as DataFormat in Pandas
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     captions = annotations['caption'].values

     return feats, captions

# From NeuralTalk
def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      # str.lower() : return a copy of the string with all the cased characters ...
      #                converted to lowercase
      # str.split(' ') : word separation
      for w in sent.lower().split(' '):
         word_counts[w] = word_counts.get(w, 0) + 1
         print word_counts[w]

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


feats, captions = get_caption_data(annotation_path, feat_path)
wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

print type(captions)
print captions.shape
print captions

print type(feats)
print feats.shape

print type(wordtoix)
print wordtoix
print type(ixtoword)
print ixtoword
print type(bias_init_vector)
print bias_init_vector