
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
from keras.models import load_model

K.set_session(sess)


width = 160
length = 120

file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/xtrain_data.npy'
x_train = np.load(file_name1)

file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/xtest_data.npy'
x_test = np.load(file_name2)

file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/ytrain_data.npy'
y_train = np.load(file_name1)

file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/ytest_data.npy'
y_test = np.load(file_name2)
print(np.shape(y_train))

#x_train = x_train[:200]
#y_train = y_train[:200]
x_train = x_train.reshape(x_train.shape[0],width ,length, 3)
x_test = x_test.reshape(x_test.shape[0], width, length, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


input_shape = (width,length,3)
num_classes = 9
batch_size = 50
epochs = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colour_160_120.h5'

cnnmodel = load_model('saved_models/colour_160_120.h5')
print("Loaded cnn model")

score = cnnmodel.evaluate(x_train, y_train, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])

score = cnnmodel.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])