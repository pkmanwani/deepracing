from __future__ import print_function
import numpy as np
import tensorflow as tf
sess = tf.Session()
import os


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import callbacks
from keras.layers import LSTM
K.set_session(sess)


num_frames = 5
batch_size = 1
epochs = 1000
num_labels = 9
file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_xtrain_data.npy'
x_train = np.load(file_name1)

file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_xtest_data.npy'
x_test = np.load(file_name2)

file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_ytrain_data.npy'
y_train = np.load(file_name1)

file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_ytest_data.npy'
y_test = np.load(file_name2)
#_train = x_train[5:]
#x_train = x_train[:11995]
y_train = y_train[5:]

#x_test = x_test[5:]
#x_test = x_test[:1995]
y_test = y_test[5:]
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))
input_shape = (6,512)

#x_train = x_train.reshape(x_train.shape[0],input_shape[0],input_shape[1])
#x_test = x_test.reshape(y_train.shape[0],input_shape[0],input_shape[1])

model = Sequential()
model.add(LSTM(100, input_shape =  input_shape))#return_sequences=False))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

#tbCallback = keras.callbacks.TensorBoard(log_dir='C:\\Users\\pratik pc\\Desktop\\cnnproj', histogram_freq=2,
#                            write_graph=True, write_images=True)
filepath2="saved_models/rnn_weights_best2.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print(np.shape(x_train))
print(np.shape(x_test))
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),callbacks = [checkpoint])


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model3.save(filepath2)