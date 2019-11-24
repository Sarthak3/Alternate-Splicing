import numpy as np 
import scipy
import h5py
from keras.layers import Embedding, recurrent, Activation, Dense, Flatten
import os
from keras import backend as K
from keras import metrics
from keras.callbacks import *
from keras.preprocessing import sequence
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout
from keras.backend.tensorflow_backend import set_session
import preprocess

from keras.preprocessing.text import one_hot


import tensorflow as tf
import matplotlib.pyplot as plt

def det_coeff(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

X_train,Y_train,X_test,Y_test = preprocess.dataz5ss()
# X_train,Y_train,X_test,Y_test = preprocess.dataz3ss()


batch_size = 512
# max_features = X_train.shape[1]
# print(max_features)
# encodedseq=[one_hot(d, 4) for d in X_train]
# print(encodedseq[0], len(encodedseq[0]))
#######################################################################################################################################
model = Sequential()
model.add(Embedding(4, 100, input_length=len(X_train[0])))
model.add(LSTM(100, return_sequences=True))
# model.add(Attention(max_features))
model.add(Flatten())
model.add(Dense(512))
# model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
# try using different optimizers and different optimizer configs
print(model.summary())
#######################################################################################################################################
# model.load_weights('model_wghts_lstm_1.h5')
model.compile(loss='mse',
              optimizer='adam',
              metrics=[det_coeff])
  

print('Train...')
history= model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.2
          )

print("Printing graph")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss_Graph.png')
plt.clf()

print("Printing graph")
plt.plot(history.history['det_coeff'])
plt.plot(history.history['val_det_coeff'])
plt.title('Model Deterministic Coefficient')
plt.ylabel('Det Coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Det_Graph.png')
plt.clf()

scores = model.evaluate(X_test, Y_test,verbose=1,batch_size=batch_size)

Y_pred=model.predict(X_test)

x=[[item[i] for item in Y_pred] for i in range(len(Y_pred[0]))]
y=[[item[i] for item in Y_test] for i in range(len(Y_test[0]))]

plt.plot(x[0], y[0], '.', color='blue')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Prediction0.png')
plt.clf()

plt.plot(x[1], y[1], '.', color='blue')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Prediction1.png')
plt.clf()

plt.plot(x[2], y[2], '.', color='blue')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Prediction2.png')
plt.clf()

plt.plot(x[3], y[3], '.', color='blue')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Prediction3.png')
plt.clf()

plt.plot(x[4], y[4], '.', color='blue')
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Prediction4.png')
plt.clf()

print("Deterministic Coefficient: {}".format(scores[1])) 

