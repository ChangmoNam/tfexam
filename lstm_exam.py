import word_data_exam # For word dataset
import os
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import pandas as pd

data_path = '/data3/flickr30k/'
img_path = os.path.join(data_path, 'feats.npy')
annotation_path = os.path.join(data_path, 'results_20130124.token') # pickle
word_count = 5

# For word dataset
#word_data_exam.word_data(data_path,img_path,annotation_path,word_count)

class Captioner():
    def __init__(self, image_dim, word_dim, hidden_dim, batch, max_len, voca_size):

        self.batch = batch
        self.image_dim = image_dim
        self.word_dim = word_dim
        self.num_steps = max_len

        self.lstm = rnn_cell.BasicLSTMCell(hidden_dim)
        self.emb_init = tf.Variable(tf.random_normal([self.image_dim,self.lstm.state_size]))
        self.initial_state = self.state = tf.zeros([self.batch, self.lstm.state_size])
        self.emb_bias = tf.Variable(tf.random_normal([self.word_dim]))
        self.embeddings = tf.Variable(tf.random_normal([voca_size, word_dim]))

    def lstm_model(self, image, words, vocabulary_size):

        image_emb = tf.matmul(image, self.emb_init)

        for i in range(self.num_steps):

            if i == 0:
                input_emb = image_emb
            else:
                input_emb = tf.nn.embedding_lookup(self.embeddings,words[:,i-1])

            output, self.state = self.lstm(input_emb[:,i],self.state)
            cost = tf.reduce_mean

        final_state = self.state


        return final_state, output

# ----- Parameters ----- #
image_dimension = 4096
embedding_size = 512
hidden_dimension = 512
batch_size = 128
# ----- Parameters ----- #


def main():

    # --------------- Data --------------- #
    img = np.load('/data3/flickr30k/feats.npy')
    word = np.load('/data3/flickr30k/word2idx.npy').tolist() # len(word) : 7734
    sentence = pd.read_table(annotation_path,sep='\t',header=None,names=['img','voca'])
    caption = sentence['voca'].values
    # --------------- Data --------------- #

    # --------------- Param --------------- #
    max_step = np.min(map(lambda x: len(x.split(' ')),caption)) # max_step = 82
    vocabulary_size = len(word) # 7734
    # --------------- Param --------------- #

    image = tf.placeholder(tf.float32, [batch_size, image_dimension])
    words = tf.placeholder(tf.float32, [batch_size, max_step])


    embeddings = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))

    captioner = Captioner(image_dimension,embedding_size,hidden_dimension,\
                          batch_size,max_step,vocabulary_size)
    final_state = captioner.lstm_model(image, words, vocabulary_size)

main()
'''
lstm = rnn_cell.BasicLSTMCell(lstm_size)
state = tf.zeros([batch_size, lstm.state_size])

loss = 0

for current_batch_of_words in words_in_dataset:

    output, state = lstm(current_batch_of_words, state)

    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)
'''