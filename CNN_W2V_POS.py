#This module will perform trigger detection using CNN

import random
random.seed(100)


from Create_Data_Model import processed_data, pad_sequences_3D, labelMatrix2OneHot, concat_2Dvectors, Flatten_3Dto2D
from Other_Utils import prob2Onehot
data = processed_data()
#from Load_Embedings import GoogleVecs
import tensorflow as tf
import numpy as np

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

#Get data
trainX, embeddings, trainY, maxLen, POS_labels, position_vectors = data.get_Data_Embeddings()
POS_vectors, _ = labelMatrix2OneHot(POS_labels)
del data
"""
print("TrainX : ", len(trainX))
print("TrainY : ", len(trainY))
print("Embd : ", len(embeddings))
print("POS : ", len(POS_labels))
print("POSit  : ", len(position_vectors))
print("Max Len : ", maxLen)
"""

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0.)
#Converting labels to binary vectors
trainY = pad_sequences(trainY, maxlen=2, value=0.)
embeddings = concat_2Dvectors(embeddings, Flatten_3Dto2D(POS_vectors))
print(embeddings[0])
print(embeddings[1])
print(embeddings[2])
print(embeddings[3])
print(embeddings[4])
print(embeddings[5])
print(embeddings[6])
print(embeddings[7])
print(embeddings[8])
print(embeddings[9])
print(embeddings[10])

# Network building
print("Beginning neural network")
net = input_data(shape=[None, maxLen])
net = embedding(net, input_dim=len(embeddings), output_dim=len(embeddings[0]), trainable=False, name="EmbeddingLayer")
print(net.get_shape().as_list())
branch1 = conv_1d(net, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(net, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(net, 128, 5, padding='valid', activation='relu', regularizer="L2")
net = merge([branch1, branch2, branch3], mode='concat', axis=1)
net = tf.expand_dims(net, 2)
net = global_max_pool(net)
net = dropout(net, 0.5)
net = fully_connected(net, 2, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.005)
print("Done neural network")

testX = trainX[int(0.3*len(trainY)):]
testY = trainY[int(0.3*len(trainY)):]

# Training
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
# Assign your own weights (for example, a numpy array [input_dim, output_dim])
model.set_weights(embeddingWeights, embeddings)
model.fit(trainX, trainY, n_epoch=3, validation_set=0.1, show_metric=True, batch_size=32, shuffle=True)
#print( model.evaluate(testX, testY) )
predictions = model.predict(testX)
predictions = prob2Onehot(predictions)
#print("Predictions : ", list(predictions[10]))



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
