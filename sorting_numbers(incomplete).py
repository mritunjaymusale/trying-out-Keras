import keras as k
from keras.layers import Dense
import random
X_train=[]
Y_train=[]
#horrible way of creating a list and sorted list
for  i in range(100):
    b=[]
    for j in range(5):
        
        a=random.randrange(0,100,step=1)
        b.append(a)
    X_train.append(b)
    Y_train.append(sorted(b))
model=k.Sequential()
model.add(Dense(units=8, activation=k.activations.relu, input_dim=5))
model.add(Dense(units=20,activation=k.activations.relu))
model.add(Dense(units=5,activation=k.activations.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

model.fit(X_train, Y_train, epochs=100,verbose=1)
model.summary()