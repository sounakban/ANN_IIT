###### This module will perform trigger detection #######

import random
random.seed(100)

from Create_Data_Model import processed_data, tagMatrix2Embeddings
from Other_Utils import prob2Onehot, pad_sequences_3D
data = processed_data()
import tensorflow as tf
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Bidirectional, Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

#Get data
trainX, word_embeddings, trainY, maxLen, POS_labels = data.get_Data_Embeddings()
POS_vectors, POS_embeddings, _ = tagMatrix2Embeddings(POS_labels)
del data

print("TrainX : ", len(trainX))
print("TrainY : ", len(trainY))
print("Word Embd : ", len(word_embeddings))
print("POS labels: ", len(POS_labels))
print("POS vecs: ", len(POS_vectors))
print("POS Embd: ", len(POS_embeddings))
print("Max Len : ", maxLen)

# Data preprocessing
## Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0)
trainY = pad_sequences_3D(trainY, maxlen=maxLen, value=[0,1])

# Defining the Network
print("Beginning neural network")

## Defining vectors and embeddings
word_inp = Input(shape=(maxLen,))
word_embed_layer = Embedding(len(word_embeddings), len(word_embeddings[0]), weights=[word_embeddings], input_length=maxLen)(word_inp)
print("Shape, word embd: ", np.shape(word_embed_layer))
POS_inp = Input(shape=(maxLen,))
POS_embed_layer = Embedding(len(POS_embeddings), len(POS_embeddings[0]), weights=[POS_embeddings], input_length=maxLen)(POS_inp)
#POS_embed_layer.set_weights(POS_embeddings)
print("Shape, POS embd: ", np.shape(POS_embed_layer))

## Combine Embeddings
embed_layer = Concatenate(axis=-1)([word_embed_layer, POS_embed_layer])
print("Shape, total embd: ", np.shape(embed_layer))

## Layer Operations
#print(net.get_shape().as_list())
#seq = Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat')(embed_layer)
#seq = Bidirectional(LSTM(256, dropout=0.5, return_sequences=True), merge_mode='concat')(embed_layer)
seq = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(embed_layer)
seq = Dropout(0.5)(seq)
mlp = TimeDistributed(Dense(2, activation='softmax'))(seq)
model = Model(inputs=[word_inp, POS_inp], outputs=mlp)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

testX = trainX[int(0.3*len(trainY)):]
testY = trainY[int(0.3*len(trainY)):]


# Training
model.fit([trainX, POS_vectors], trainY, epochs=3, validation_split=0.1, verbose=2, batch_size=32, shuffle=True)
#print( model.evaluate(testX, testY) )
predictions = model.predict(testX)
predictions = prob2Onehot(predictions)
#print("Predictions : ", list(predictions[10]))


"""
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
