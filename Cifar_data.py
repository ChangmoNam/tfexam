import cPickle
import numpy as np



def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def next_batch(data,label,i,batch_size=10):
    data = data[i*batch_size:(i+1)*batch_size,:]
    label = label[i*batch_size:(i+1)*batch_size,:]
    return data,label

def label2onehot(label,batch_size):
    label_out = np.zeros((batch_size,10))
    for i in range(0,batch_size,1):
        label_out[i,label[i]] = 1
    return label_out

def data_converge():
    cifar10 = unpickle('/data3/cifar10/cifar-10-batches-py/data_batch_1')
    data = cifar10['data']
    label = np.array(cifar10['labels']).reshape((10000,1))
    for i in range(2,6,1):
        cifar10 = unpickle('/data3/cifar10/cifar-10-batches-py/data_batch_'+str(i))
        data = np.append(data,cifar10['data'],axis=0)
        label = np.append(label,np.array(cifar10['labels']).reshape((10000,1)),axis=0)

    return data, label

def input_data(i, batch_size):

    #data = unpickle('/data3/cifar10/cifar-10-batches-py/data_batch_1')

    #data_val = data['data'] # np.ndarray / dtype=unit8
    #label_val = data['labels'] # list
    #label_val = np.array(label_val).reshape((10000,1))

    data_val, label_val = data_converge()

    #print data_val.shape
    #print label_val.shape
    #print type(label_val)
    data_value, label = next_batch(data=data_val,label=label_val,\
                                   i=i,batch_size=batch_size)

    label_value = label2onehot(label=label, batch_size=batch_size)

    #label = np.array(label)
    #print data_value.shape
    #print label.shape
    #label_value = np.zeros((batch_size,10))
    #for i in range(0,batch_size,1):
        #label_value = np.zeros((batch_size,10))
        #label_value[i,label[i]] = 1
     #   print label[i]
     #   print label_value
    return data_value, label_value
###

#for i in range(0,10,1):
   # c = b[i]
   # k = np.zeros((1,10))
   # print k
  #  print k[:,c]
  #  k[:,c] = 1
  #  print i
 #   print c
#    print k
#    #print z

###c = np.array(b)
#d = c.reshape((1,10))
#e = d[0,:]
#e = d.reshape((-1,10))
#print type(c)
#print c
#print d
#print d.shape
#print e.shape

#x = np.arange(3)
#print type(x)
#print x.shape
#print x
#c = b
#print b
#print type(b)
#print c
#print type(c)
#print c.shape

#c = np.reshape(c,[1,10])
#print c.shape
#print type(c)
