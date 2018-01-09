#This module will perform trigger detection using CNN

import random
random.seed(100)

from Create_Data_Model import processed_data, pad_sequences_3D, labelMatrix2OneHot, concat_2Dvectors, Flatten_3Dto2D
from Other_Utils import prob2Onehot
data = processed_data()
import tensorflow as tf
import numpy as np

from tflearn import DNN, get_layer_variables_by_name
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.conv import conv_1d, global_max_pool, global_avg_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

#Get data
trainX, embeddings, trainY, maxLen, POS_labels = data.get_Data_Embeddings()
POS_vectors, _ = labelMatrix2OneHot(POS_labels)
del data

print("TrainX : ", len(trainX))
print("TrainY : ", len(trainY))
print("Embd : ", len(embeddings))
print("POS : ", len(POS_labels))
print("Max Len : ", maxLen)


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0.)
#Converting labels to binary vectors
trainY = pad_sequences(trainY, maxlen=2, value=0.)
#Concatenate POS tags to the embeddings
#embeddings = concat_2Dvectors(embeddings, Flatten_3Dto2D(POS_vectors))
POS_vectors = Flatten_3Dto2D(POS_vectors)
print(len(POS_vectors))
print(len(POS_vectors[0]))

"""
print(embeddings[0][300:])
print(embeddings[1][300:])
print(embeddings[2][300:])
print(embeddings[3][300:])
print(embeddings[4][300:])
print(embeddings[5][300:])
print(embeddings[6][300:])
print(embeddings[7][300:])
print(embeddings[8][300:])
print(embeddings[9][300:])
print(embeddings[10][300:])
#"""

# Network Description
print("Beginning neural network")
net = input_data(shape=[None, maxLen])
emb1 = embedding(net, input_dim=len(embeddings), output_dim=len(embeddings[0]), trainable=False, name="EmbeddingLayer")
emb2 = embedding(net, input_dim=len(POS_vectors), output_dim=len(POS_vectors[0]), trainable=False, name="POSLayer")
net = tf.concat([emb1, emb2], 2)
#net = merge([branch1, branch2, branch3, branch4], mode='concat', axis=1)
print("Shape after embeddings : ", net.get_shape().as_list())
branch1 = conv_1d(net, 150, 2, padding='valid', activation='relu', regularizer="L2")
print("Shape after CNN1 : ", branch1.get_shape().as_list())
branch2 = conv_1d(net, 150, 3, padding='valid', activation='relu', regularizer="L2")
print("Shape after CNN2 : ", branch2.get_shape().as_list())
branch3 = conv_1d(net, 150, 4, padding='valid', activation='relu', regularizer="L2")
print("Shape after CNN3 : ", branch3.get_shape().as_list())
branch4 = conv_1d(net, 150, 5, padding='valid', activation='relu', regularizer="L2")
print("Shape after CNN4 : ", branch4.get_shape().as_list())
branch1 = tf.expand_dims(branch1, 2)
branch1 = global_avg_pool(branch1)
print("Shape after pooling : ", branch1.get_shape().as_list())
branch2 = tf.expand_dims(branch2, 2)
branch2 = global_avg_pool(branch2)
print("Shape after pooling : ", branch2.get_shape().as_list())
branch3 = tf.expand_dims(branch3, 2)
branch3 = global_avg_pool(branch3)
print("Shape after pooling : ", branch3.get_shape().as_list())
branch4 = tf.expand_dims(branch4, 2)
branch4 = global_avg_pool(branch4)
print("Shape after pooling : ", branch4.get_shape().as_list())
net = merge([branch1, branch2, branch3, branch4], mode='concat', axis=1)
print("Shape after CNN merging : ", net.get_shape().as_list())
net = dropout(net, 0.5)
net = fully_connected(net, 1024, activation='softmax')
print("Shape after 1024 fully connected layer : ", net.get_shape().as_list())
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.005)
print("Done neural network")

testX = trainX[int(0.3*len(trainY)):]
testY = trainY[int(0.3*len(trainY)):]

# Training
model = DNN(net, clip_gradients=0., tensorboard_verbose=2)
embeddingWeights = get_layer_variables_by_name('EmbeddingLayer')[0]
POSWeights = get_layer_variables_by_name('POSLayer')[0]
#! Assign your own weights (for example, a numpy array [input_dim, output_dim])
model.set_weights(embeddingWeights, embeddings)
model.set_weights(POSWeights, POS_vectors)
model.fit(trainX, trainY, n_epoch=3, validation_set=0.1, show_metric=True, batch_size=50, shuffle=True)
#print( model.evaluate(testX, testY) )
predictions = model.predict(testX)
predictions = prob2Onehot(predictions)
#print("Predictions : ", list(predictions[10]))
print("Number of triggers : ", len([pred for pred in testY if list(pred)==[1,0]]))



##Calculate F1 Score
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(predictions.shape[0]):
    if list(testY[i]) == [1,0]:
        if list(predictions[i]) == [1,0]:
            tp += 1
        else:
            fn += 1
    else:
        if list(predictions[i]) == [1,0]:
            fp += 1
        else:
            tn += 1


print(predictions.shape)
print(testX.shape)
print(testY.shape)
print("Tru-Pos : ", tp)
print("Tru-Neg : ", tn)
print("Fals-Pos : ", fp)
print("Fals-Neg : ", fn)

pr = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = 2*((pr*rec)/(pr+rec))
print("Precision : ", pr)
print("Recall : ", rec)
print("F1 : ", f1)
