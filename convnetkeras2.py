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
x_train = x_train.reshape(x_train.shape[0],length,width, 3)
x_test = x_test.reshape(x_test.shape[0], length, width, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


input_shape = (length,width, 3)
num_classes = 9
batch_size = 20
epochs = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colour_160_120_best.h5'




model = Sequential()

model.add(Conv2D(64, (5,5), padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0002, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

#tbCallback = keras.callbacks.TensorBoard(log_dir='C:\\Users\\pratik pc\\Desktop\\cnnproj', histogram_freq=2,
#                            write_graph=True, write_images=True)
filepath2="weights_best_new.h5"
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

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)






