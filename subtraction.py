import keras as k
from keras.layers import Dense
import random
X=[]
Y=[]
for i in range(100):
    temp=random.randint(0,100)
    X.append([temp])
    Y.append([temp-2])

print(X)
print(Y)
model= k.Sequential()
model.add(Dense(units=1,activation=k.activations.relu,input_dim=1))
model.compile(loss=k.losses.sparse_categorical_crossentropy,
              optimizer='adam')
model.summary()

model.fit(X,Y,epochs=100)
print(model.predict([[5],[4]]))
#loss to damn high 