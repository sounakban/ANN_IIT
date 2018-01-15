
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge


# In[2]:

import word_based_train_file_fire_2017


# In[3]:

len_vocab=word_based_train_file_fire_2017.return_data()
print(len_vocab)


# In[5]:

train_x=word_based_train_file_fire_2017.x_train_index
train_y=word_based_train_file_fire_2017.y_train
max_sentence_length=0
i=0
c=0
l=0
while i< len(train_x):
    l=len(train_x[i])
    if l> max_sentence_length:
        c=i
        max_sentence_length=l
    i+=1
print(max_sentence_length)

# In[6]:


x_train=sequence.pad_sequences(train_x, maxlen=max_sentence_length, padding='post', value=len_vocab)
y_train=sequence.pad_sequences(train_y, maxlen=max_sentence_length, padding='post', value=0)


# In[7]:

test_x=word_based_train_file_fire_2017.x_test_index
x_test=sequence.pad_sequences(test_x, maxlen=max_sentence_length, padding='post', value=len_vocab)


# In[8]:

len(x_train[0])


# In[9]:

y_train[1021]


# In[10]:

y_train_hot=[]
i=0
while i< len(y_train):
    h=np.eye(2)[y_train[i]]
    h=h.tolist()
    y_train_hot.append(h)
    i+=1


# In[11]:

np.shape(x_train)
np.shape(y_train_hot)


# In[14]:

np.random.seed(1337)  # for reproducibility
max_features = len_vocab+1#vocab size
batch_size = 40#batch size
maxlen = max_sentence_length#max tweet_characterized length
hidden=60#size of hidden layer
nb_classes=2
filter_sizes=[2,3,4]
num_filters=30


# In[15]:

import keras
sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(max_features, 50, input_length=maxlen)(sequence)
print(np.shape(embedded))
embedded1= keras.layers.convolutional.Conv1D(filters=30, kernel_size=[3], strides=1, padding='same')(embedded)
print(np.shape(embedded1))
'''convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='same',
                         activation='relu',
                         subsample_length=1)(embedded)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
print(convs)
'''
embedded2= keras.layers.convolutional.Conv1D(filters=20, kernel_size=[4], strides=1, padding='same')(embedded1)
embed3=keras.layers.wrappers.TimeDistributed(embedded2)
print(np.shape(embedded2))
forwards = LSTM(output_dim=hidden, return_sequences=True)(embedded2)
backwards = LSTM(output_dim=hidden, return_sequences=True, go_backwards=True)(embedded2)
merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
print(np.shape(merged))
#forwards1 = LSTM(output_dim=hidden, return_sequences=True)(merged)
#backwards1 = LSTM(output_dim=hidden, return_sequences=True, go_backwards=True)(merged)
#merged1 = merge([forwards1, backwards1], mode='concat', concat_axis=-1)
after_dp = Dropout(0.5)(merged)
output = keras.layers.wrappers.TimeDistributed(Dense(output_dim=nb_classes, activation='softmax'))(after_dp)
print(np.shape(output))
model = Model(input=sequence, output=output)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
y_train_hot=np.array(y_train_hot)
model.fit(x_train, y_train_hot,batch_size=batch_size,epochs=40,validation_split=0.2)
y_pred=model.predict(x_test)


# In[18]:

#print(np.shape(y_pred))


# In[16]:

output_final=np.argmax(y_pred,axis=-1)


# In[17]:

final_result=output_final.tolist()


# In[23]:




# In[ ]:
