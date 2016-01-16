# import word_data_exam ::: For word dataset
import os

data_path = '/data3/flickr30k/'
img_path = os.path.join(data_path, 'feats.npy')
annotation_path = os.path.join(data_path, 'results_20130124.token') # pickle

# For word dataset
#word_data_exam.word_data(data_path,img_path,annotation_path)