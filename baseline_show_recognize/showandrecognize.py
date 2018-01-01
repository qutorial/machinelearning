#!/usr/bin/env python3
# source: https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model

import argparse
parser = argparse.ArgumentParser(description='Recognizing hand written mnist numbers.')
parser.add_argument('n', metavar='n', type=int, help='number of the sample to recognize and show')
args = parser.parse_args()
n = args.n

# fix random seed for reproducibility
seed = 223
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


def train_model():
  # build the model
  model = baseline_model()
  # Fit the model
  model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
  # Final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Baseline Error: %.2f%%" % (100-scores[1]*100))

  print("Training finished");

  model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

  return model

try:
  model = load_model('my_model.h5')
  print("Model loaded successfully!")
except:
  print("No saved model, training one...")
  model = train_model()



print("Recongnizing number in the sample #%d" % n)
print("It is %d" % model.predict_classes(X_test[n:n+1])[0] )


import matplotlib.pyplot as plt
plt.subplot(111)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_test[n], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
