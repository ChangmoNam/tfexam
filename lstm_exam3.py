import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle

from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
#from cnn_util import *



def get_caption_data(annotation_path, feat_path):
     feats = np.load(feat_path)
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     captions = annotations['caption'].values

     return feats, captions

def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
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



def lstm_model(_weights, _biases, _Wemb, _config):

    _image = tf.placeholder(tf.float32, [_config.batch_size, _config.dim_image])
    _sentence = tf.placeholder(tf.int32, [_config.batch_size, _config.maxlen+2])
    _mask = tf.placeholder(tf.float32, [_config.batch_size, _config.maxlen+2])

    lstm = rnn_cell.BasicLSTMCell(_config.dim_hidden)
    image_emb = tf.matmul(_image, _weights['encoding_img_W']) + _biases['encoding_img_b'] # (batch_size, dim_hidden)

    state = tf.zeros([_config.batch_size, lstm.state_size])

    _loss = 0.0

    with tf.variable_scope("RNN"):
        for i in range(_config.maxlen+2): # maxlen + 1
            if i == 0:
                current_emb = image_emb
            else:
                with tf.device("/cpu:0"):
                    current_emb = tf.nn.embedding_lookup(_Wemb, _sentence[:,i-1]) + _biases['bemb']

            if i > 0 : tf.get_variable_scope().reuse_variables()

            output, state = lstm(current_emb, state) # (batch_size, dim_hidden)

            if i > 0:
                labels = tf.expand_dims(_sentence[:, i], 1) # (batch_size)
                indices = tf.expand_dims(tf.range(0, _config.batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(
                        concated, tf.pack([_config.batch_size, _config.n_words]), 1.0, 0.0) # (batch_size, n_words)

                logit_words = tf.matmul(output, _weights['embed_word_W']) + _biases['embed_word_b'] # (batch_size, n_words)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * _mask[:,i]#tf.expand_dims(mask, 1)

                current_loss = tf.reduce_sum(cross_entropy)
                _loss = _loss + current_loss

        _loss = _loss / tf.reduce_sum(_mask[:,1:])

        return _loss, _image, _sentence, _mask


#learning_rate = 0.001
n_epochs = 1000
###############################################################

model_path = './models/tensorflow'
vgg_path = './data/vgg16.tfmodel'
data_path = '/data3/flickr30k/'
feat_path = '/data3/flickr30k/feats.npy'
annotation_path = os.path.join(data_path, 'results_20130124.token')
################################################################

class get_config(object):
    dim_embed = 256
    dim_hidden = 256
    dim_image = 4096
    batch_size = 128
    maxlen = 0


def get_variables(config):

    with tf.device("/cpu:0"):
        Wemb = tf.Variable(tf.random_normal([config.n_words, config.dim_embed]))
    encoding_img_W = tf.Variable(tf.random_normal([config.dim_image, config.dim_hidden]))
    embed_word_W = tf.Variable(tf.random_normal([config.dim_hidden, config.n_words]))

    bemb = tf.Variable(tf.zeros([config.dim_embed]))
    encoding_img_b = tf.Variable(tf.zeros([config.dim_hidden]))
    embed_word_b = tf.Variable(tf.zeros([config.n_words]))
    #'embed_word_b': tf.Variable(bias_init_vector.astype(np.float32))}


def train():
    #sess = tf.InteractiveSession()
    learning_rate = 0.001
    momentum = 0.9
    feats, captions = get_caption_data(annotation_path, feat_path)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    #np.save('data/ixtoword', ixtoword)
    config = get_config()
    #variables = get_variables(config)
    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]

    config.n_words = len(wordtoix)
    config.maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
    #variables = get_variables(config)

    #image = tf.placeholder(tf.float32, [batch_size, dim_image])
    #sentence = tf.placeholder(tf.int32, [batch_size, maxlen+2])
    #mask = tf.placeholder(tf.float32, [batch_size, maxlen+2])

    with tf.device("/cpu:0"):
        Wemb = tf.Variable(tf.random_normal([config.n_words, config.dim_embed]))

    weights = {#'Wemb': tf.Variable(tf.random_normal([config.n_words, config.dim_embed])),
               'encoding_img_W': tf.Variable(tf.random_normal([config.dim_image, config.dim_hidden])),
               'embed_word_W': tf.Variable(tf.random_normal([config.dim_hidden, config.n_words]))}

    biases = {'bemb': tf.Variable(tf.zeros([config.dim_embed])),
              'encoding_img_b': tf.Variable(tf.zeros([config.dim_hidden])),
              #'embed_word_b': tf.Variable(tf.zeros([config.n_words]))}
              'embed_word_b': tf.Variable(bias_init_vector.astype(np.float32))}


    sess = tf.InteractiveSession()

    loss, image, sentence, mask = lstm_model(weights, biases, Wemb, config)

    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    for epoch in range(n_epochs):
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        for start, end in zip( \
                range(0, len(feats), config.batch_size),
                range(config.batch_size, len(feats), config.batch_size)
                ):
            print (start,end)
            current_feats = feats[start:end]
            current_captions = captions[start:end]

            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=config.maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_caption_matrix ))
            #  +2 -> #START# and '.'

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats,
                sentence : current_caption_matrix,
                mask : current_mask_matrix
                })

            print "Current Cost: ", loss_value

        print "Epoch ", epoch, " is done. Saving the model ... "
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        learning_rate *= 0.95





train()