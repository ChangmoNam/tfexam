import scipy.io as sio
import tensorflow as tf
import Cifar_data
import numpy as np

batch_size = 100
training_iters = 100000
n_input = 3072
#n_input = 1024
n_classes = 10
IMG_SIZE = 32
learning_rate = 0.001
display_step = 1
dropout = 0.5

mat = sio.loadmat('/home/ncm4114/PycharmProjects/example/imagenet-vgg-f.mat')

layers = mat['layers'][0,0]['weights'][0,0][0,0]#[:,:,:,1]

x = tf.placeholder(tf.float32, shape=(batch_size, n_input))
y = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
keep_prob = tf.placeholder(tf.float32) #drop_out

def convert2tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv2d(img,w,b):
    #img = tf.reshape(img, shape=[-1,32,32,3])
    return tf.nn.relu(tf.nn.bias_add\
                          (tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))

def conv_net(x,w,b,drop_out):
    x = tf.reshape(x, shape=[-1,IMG_SIZE,IMG_SIZE,3])
    conv1 = conv2d(x,w['conv1'],b)
    pool1 = max_pooling(conv1)

    #Fully Connected Layer
    dense1 = tf.reshape(pool1, [-1, w['fc1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, w['fc1']))
    dense1 = tf.nn.dropout(dense1,drop_out)

    out = tf.matmul(dense1,w['fc2'])
    return out


weight1 = convert2tensor(mat['layers'][0,0]['weights'][0,0][0,0]) #tensor
bias1 = convert2tensor(mat['layers'][0,0]['weights'][0,0][0,1].flatten())
weight_fc1 = tf.Variable(tf.random_normal([11*11*64, 1000]))
weigth_fc2 = tf.Variable(tf.random_normal([1000,n_classes]))

weight = {
    'conv1': convert2tensor(mat['layers'][0,0]['weights'][0,0][0,0]),
    'fc1': tf.Variable(tf.random_normal([16*16*64, 1000])),
    'fc2': tf.Variable(tf.random_normal([1000,n_classes]))
}
#print weight['fc1'].get_shape().as_list()[0]
conv_out = conv_net(x,weight,bias1,keep_prob)

#loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(conv_out,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(conv_out,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    step = 1
    sess.run(init)
    while step*batch_size < training_iters:
        batch_xs, batch_ys = Cifar_data.input_data(i=step,batch_size=batch_size)
        print batch_xs.shape
        print batch_ys.shape
        sess.run(optimizer, feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step%display_step==0:
            acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys,keep_prob:1})
            loss = sess.run(cost, feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

        step += 1
