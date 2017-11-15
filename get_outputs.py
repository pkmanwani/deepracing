import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from keras.models import load_model
import random


rnn = True
width = 160
length = 120
last_choice = 1
colour = False
#get outputs
cnnmodel = load_model('saved_models/weights_best_new.h5')
print("Loaded cnn model")
total = 18
layerno = 10
for i in range(0,total - layerno):
	cnnmodel.pop()
last_time = time.time()
screen = grab_screen(region = (0,23, 800, 623))
screen = screen[:,:,:3]
#cv2.imwrite('image.jpg', screen)
#screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
screen = cv2.resize(screen,(width,length),3)
print(np.shape(screen))
#inputs = screen.flatten()
inputs  = screen.reshape(1,length,width,3)
print(np.shape(screen))
inputs = inputs.astype('float32')
cnn_out = cnnmodel.predict(inputs)
a=np.shape(cnn_out)[1]
b=np.shape(cnn_out)[2]
c=np.shape(cnn_out)[3]
cnn_out = np.reshape(cnn_out, (a,b,c))
print(np.shape(cnn_out))
#cnn_out = list(cnn_out)
c = 0
for i in range(cnn_out.shape[-1]):
	features = cnn_out[...,i]
	print(np.shape(features))
	cv2.imwrite('data/training/track1/features/feature-{0}'.format(c) +'.jpg', features)
	c += 1








