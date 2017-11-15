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
model_name = 'colour_160_120_nvidia.h5'
#number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Initiating the model
model = Sequential()
# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Conv2D(nb_filters1, kernel_size[0], kernel_size[1],
	border_mode='valid',input_shape=input_shape))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Conv2D(nb_filters2, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Conv2D(nb_filters3, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Conv2D(nb_filters4, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
# Apply dropout of 25%
model.add(Dropout(0.25))
# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(num_classes))
# Print out summary of the model
model.summary()
# Compile model using Adam optimizer
# and loss computed by mean squared error

opt = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
filepath2='saved_models/colour_160_120_nvidia_best.h5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

### Model training
history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=epochs,
                    verbose=1, validation_data=(x_test, y_test),callbacks = [checkpoint])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
