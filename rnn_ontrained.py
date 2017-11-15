import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from keras.models import load_model
import random
import os
import tensorflow as tf
sess = tf.Session()
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import callbacks
K.set_session(sess)

num_frames = 5
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
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'rnn_colour_160_120.h5' 





def create_rnn_dataset(dataset, num_frames = 10):
	model = load_model('saved_models/colour_160_120.h5')
	print("Loaded model")
	dataX = []
	dataY = []
	count =0
	#print(np.shape(dataset))
	#x = int((np.shape(dataset)[0]/1000))
	#print(x)
	#for thousands in range(x-1):
	model.pop()
	model.pop()
	model.pop()
	rnn_dataset = model.predict(dataset)
	print(np.shape(rnn_dataset))
	#rnn_dataset = np.vstack((np.zeros((num_frames,512)), rnn_dataset))
	#print(np.shape(rnn_dataset))
	for i in range(len(rnn_dataset)-num_frames):
		a = rnn_dataset[i:(i+num_frames+1)]
		#print(np.shape(a))
		dataX.append(a)
	#print(np.shape(dataX))
	return np.array(dataX)
x_train = create_rnn_dataset(x_train, num_frames)
y_train = y_train
x_test = create_rnn_dataset(x_test, num_frames)
y_test= y_test
print(np.shape(x_train))

file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_xtrain_data.npy'
np.save(file_name1, x_train)
file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_xtest_data.npy'
np.save(file_name2, x_test)
file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_ytrain_data.npy'
np.save(file_name1, y_train)
file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/rcnn_ytest_data.npy'
np.save(file_name2, y_test)

#model3 = Sequential()

#model3.add(LSTM(10, input_shape = (num_frames,512),return_sequences = True)