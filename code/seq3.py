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
import visual
from util import *
from keras.preprocessing.text import one_hot


import tensorflow as tf
import matplotlib.pyplot as plt

# X_train,Y_train,X_test,Y_test = preprocess.dataz5ss()
X_train,Y_train,X_test,Y_test = preprocess.dataz3ss()


batch_size = 512

model = Sequential()
model.add(Embedding(5, 100, input_length=len(X_train[0])))
model.add(LSTM(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(512))
# model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
# try using different optimizers and different optimizer configs
print(model.summary())

model.compile(loss='mse',
              optimizer='adam',
              metrics=[det_coeff])
  

print('Train...')
print(X_test.shape)
history= model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.2
          )

# visual.graph_epoch(history)

# scores = model.evaluate(X_test, Y_test,verbose=1,batch_size=batch_size)

winlen=5

def printpatseq(j, numseq):
	pat=""
	seq=""
	for i in range(len(numseq)):
		c=""
		if (numseq[i]==0.0):
			c="A"
		elif (numseq[i]==1.0):
			c="T"
		elif (numseq[i]==2.0):
			c="G"
		elif (numseq[i]==3.0):
			c="C"
		seq+=c
		if (i<=j) and (i>j-winlen):
			pat+=c
	return pat, seq

for i in range(len(X_test)):
	x=np.copy(X_test[i])
	yact=Y_test[i]
	ypred0=model.predict(x.reshape(1, len(x)))[0]
	for j in range(winlen):
		x[j]=4.0
	ypred1=model.predict(x.reshape(1, len(x)))[0]
	ssd=((ypred1-ypred0)**2).sum()
	j=winlen-1
	if (ssd>1):
		print(ssd, printpatseq(j, X_test[i]))
	j+=1
	while(j<len(x)):
		x[j-winlen]=X_test[i][j-winlen]
		x[j]=4.0
		ypred1=model.predict(x.reshape(1, len(x)))[0]
		ssd=((ypred1-ypred0)**2).sum()
		if (ssd>1):
			print(ssd, printpatseq(j, X_test[i]))
		j+=1

	

# Y_pred=model.predict(X_test)

# visual.pred_test_visual(Y_pred, Y_test, scores)

