#This module will perform trigger detection

from Create_Data_Model import processed_data
data = processed_data()
import tensorflow as tf

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

#Get data
trainX, trainY, maxLen, vocabSize, _ = data.get_Data_Vectors()

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0.)
#Converting labels to binary vectors
trainY = pad_sequences(trainY, maxlen=maxLen, value=0.)

# Network building
print("Beginning neural network")
net = input_data(shape=[None, maxLen])
net = embedding(net, input_dim=vocabSize, output_dim=128)
#print(net.get_shape().as_list())
net = bidirectional_rnn(net, BasicLSTMCell(512), BasicLSTMCell(512))
net = dropout(net, 0.5)
net = fully_connected(net, maxLen, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')#, learning_rate=0.001)


#"""
# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.2, show_metric=True, batch_size=64, shuffle=True)
#"""
