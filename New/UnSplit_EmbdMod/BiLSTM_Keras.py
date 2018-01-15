#This module will perform trigger detection

import random
random.seed(100)

from Create_Data_Model import processed_data, pad_sequences_3D, labelMatrix2OneHot, concat_2Dvectors, Flatten_3Dto2D
from Other_Utils import prob2Onehot
data = processed_data()
import tensorflow as tf
import numpy as np

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Bidirectional
from keras.optimizers import Adam

#Get data
trainX, trained_embeddings, trainY, maxLen, POS_labels = data.get_Data_Embeddings()
POS_vectors, POS_emdeddings, _ = tagMatrix2Embeddings(POS_labels)
del data

print("TrainX : ", len(trainX))
print("TrainY : ", len(trainY))
print("Embd : ", len(trained_embeddings))
print("POS : ", len(POS_labels))
print("Max Len : ", maxLen)

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0)
#Converting labels to binary vectors
trainY = pad_sequences_3D(trainY, maxlen=maxLen, value=[0,1])
#trained_embeddings = concat_2Dvectors(trained_embeddings, Flatten_3Dto2D(POS_vectors))

# Network building
print("Beginning neural network")
inp = Input(shape=(maxlen,))
embed_layer = Embedding(len(trained_embeddings), len(trained_embeddings[0]), input_length=maxlen)(net)
embed_layer.set_weights(trained_embeddings)
#print(net.get_shape().as_list())
#seq = Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat')(embed_layer)
#seq = Bidirectional(LSTM(256, dropout=0.5, return_sequences=True), merge_mode='concat')(embed_layer)
seq = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(embed_layer)
seq = Dropout(0.5)(seq)
mlp = TimeDistributed(Dense(2, activation='softmax'))(seq)
net = Model(input=inp, output=mlp)
optimizer = Adam(lr=0.001)
net.compile(optimizer=optimizer, loss='categorical_crossentropy')

testX = trainX[int(0.3*len(trainY)):]
testY = trainY[int(0.3*len(trainY)):]

"""
# Training
model.fit(trainX, trainY, epochs=3, validation_split=0.1, verbose=2, batch_size=32, shuffle=True)
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
"""
