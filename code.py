import tensorflow as tf
import numpy as np
from tensorflow import keras as ks
import random as rn
models = tf.keras.models
layers = tf.keras.layers
model_nn = models.Sequential()

model_nn.add(layers.InputLayer(input_shape=(3)))

model_nn.add(layers.Dense(3,activation='sigmoid'))
model_nn.add(layers.Dense(4,activation='sigmoid'))
model_nn.add(layers.Dense(2,activation='sigmoid'))

model_nn.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
model_nn.summary()
inputs = np.array([[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[1,1,1],[1,1,0]])
target_output = np.array([[0,1], [0,0], [1,1], [1,0],[0,1], [0,0], [1,1], [1,0]])
print('NN- input shape: ', model_nn.input_shape)
print('NN- output shape: ',target_output.shape)
inputs_train =[]
outputs_train = []
for i in range(100):
    r = rn.randint(0, 3)
    inputs_train.append(inputs[r])
    outputs_train.append(target_output[r])
print('Length of input_train: ',(len(inputs_train)))
model_nn.fit(np.array(inputs_train), np.array(outputs_train), epochs = 10)
model_nn.predict(np.array([[1.7,0.1,0.1], [6.3,0.9,0.8], [9.4,0.1,0.9]]))