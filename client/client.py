#!/usr/bin/env python
# coding: utf-8

# In[1]:


users = 1

C = 1
E = 1
B = 10 # 'all' for a single minibatch

rounds = 10 # default
local_epochs = 1 # default
lr = 0.1

# loss='categorical_crossentropy'
# metrics = ['accuracy']
# cce = tf.keras.losses.CategoricalCrossentropy()


# In[2]:


import logging
import math
import os
import pickle
import random
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import timedelta
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
tf.get_logger().setLevel(logging.ERROR)

import torch

import h5py
import socket
import struct
import pickle


# In[3]:


client_order = int(input("client_order(start from 0): "))


# In[4]:


num_traindata = 6000


# ## Data load

# In[5]:


mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[6]:


def noniid_partition(y_train):
  """
    sort the data by digit label, divide it into 20 shards of size 3000, and assign each of 10 clients 2 shards.
  """
  n_shards = 20
  n_per_shard = 3000

  indexes_per_client = {}
  indexes = y_train.argsort()

  indexes_shard = np.arange(0, n_shards)
  random.shuffle(indexes_shard)
  
  start_idx_shard_1 = indexes_shard[client_order*2]*n_per_shard
  start_idx_shard_2 = indexes_shard[client_order*2+1]*n_per_shard
  indexes_per_client[client_order] = np.concatenate((indexes[start_idx_shard_1:start_idx_shard_1+n_per_shard],
                                            indexes[start_idx_shard_2:start_idx_shard_2+n_per_shard]))
    
  return indexes_per_client


# In[7]:


indexes_per_client = noniid_partition(y_train)


# In[8]:


print(indexes_per_client)


# Normalize, Expand Dims, and Transform Labels

# In[9]:


X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")


# ### Create Batched Dataset

# In[10]:


def create_batch(indexes_client, X_train, y_train, B):
    x = []
    y = []    
    for i in indexes_client:
      x.append(X_train[i])
      y.append(y_train[i])

    dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))
    return dataset.shuffle(len(y)).batch(len(y_train) if B=='all' else B)


# In[11]:


client_dataset_batched = {}
for i, indexes in tqdm(indexes_per_client.items()):
  client_dataset_batched[i] = create_batch(indexes, X_train, y_train, B)

train_batched = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(len(y_train)) # for testing on train set
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


# ## model

# In[12]:


class CNN:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, padding='same', kernel_size=(5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        return model


# In[13]:


model = CNN()
local_model = model.build((28,28,1))
local_model.summary()


# In[14]:


rounds = 100 # default
local_epochs = 1 # default
lr = 0.1


# ## Socket initialization
# ### Required socket functions

# In[15]:


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


# ### Set host address and port number

# In[16]:


host_name = input("IP address: ")
port_number = 12345
max_recv = 100000


# ### Open the client socket

# In[17]:


s = socket.socket()
s.connect((host_name, port_number))


# ## SET TIMER

# In[18]:


start_time = time.time()    # store start time
print("timmer start!")


# In[19]:


msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
send_msg(s, len(train_batched))


# In[20]:


loss='categorical_crossentropy'
metrics = ['accuracy']
cce = tf.keras.losses.CategoricalCrossentropy()


# In[21]:


# global_weights = recv_msg(s)
# local_model.set_weights(global_weights)

# optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
# local_model.compile(loss=loss,
#                     optimizer=optimizer,
#                     metrics=metrics)
# history = local_model.fit(client_dataset_batched[client_order], epochs=E, verbose=1)
# evaluate = local_model.evaluate(test_batched)

# msg = evaluate
# send_msg(s, msg)

# print('Finished Training')


# In[22]:


# update weights from server
# train
for r in range(rounds):  # loop over the dataset multiple times
    global_weights = recv_msg(s)
#     local_model.load_weights(global_weights)
    local_model.set_weights(global_weights)

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
    local_model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=metrics)
    history = local_model.fit(client_dataset_batched[client_order], epochs=E, verbose=1)
    evaluate = local_model.evaluate(test_batched)

    msg = local_model.get_weights()
    send_msg(s, msg)

print('Finished Training')


# In[ ]:


end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))


# In[ ]:




