import pandas as pd
import keras as K
from keras.layers import Dense

import numpy as np
import time
#getting the data from file 
data = np.genfromtxt("breast-cancer-wisconsin.data", delimiter=",", missing_values="999", filling_values=1.)

#organizing the data taken form https://github.com/ejmejm/Breast-Cancer-NN
#Separate the X
dataX = data[:, 1:-1]
#Separate the Y
pre_dataY = data[:, -1]
#Convert the Y to one hot
dataY = np.zeros((pre_dataY.size, 2))
for i in range(len(pre_dataY)):
    if pre_dataY[i] == 2:
        dataY[i][0] = 1
    else:
        dataY[i][1] = 1
#organzing completed



# keras model
model = K.models.Sequential()
model.add(K.layers.Dense(units=9,activation=K.activations.relu,input_shape=(9,)))
model.add(K.layers.Dropout(0.2))
model.add(Dense(units=10,activation=K.activations.relu))
model.add(K.layers.Dropout(0.2))
model.add(Dense(units=7,activation=K.activations.relu))
model.add(K.layers.Dropout(0.2))
model.add(Dense(units=2,activation=K.activations.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

start=time.time()
model.fit(dataX,dataY,epochs=300,verbose=0)
print(time.time()-start)