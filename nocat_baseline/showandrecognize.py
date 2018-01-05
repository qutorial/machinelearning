#!/usr/bin/env python3
# source: https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
import sys

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

vals = np.random.randint(1, 11, 100)
labels = (vals > 5).astype(int)

# load data
(X_train, y_train), (X_test, y_test) = (vals[:80], labels[:80]), \
(vals[80:], labels[80:])

num_pixels = 1
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 10
X_test = X_test / 10

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(16, input_dim=1, activation='relu'))
  model.add(Dense(2, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  return model

# build the model
model = baseline_model()
# Fit the model
history = model.fit(X_train, y_train, \
  validation_data=(X_test, y_test), epochs=3, \
  batch_size=30, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print("Predictions: \n", np.rint(model.predict( np.array( [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]] ) / 10 ) ) )


unspreds = model.predict( np.array( [[1.5],[20],[-1],[0],[11],[10.5],[4.5],[6.7],[8.9],[0.3]] ) / 10 )
unspreds = np.rint(unspreds * 100)

print("Unseen predictions: \n",  unspreds) 


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['acc']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()