import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os
import cv2
import time
keyw = []
keys = []
keya = []
keyd = []
keywa = []
keywd = []
keysa = []
keysd = []
keynk= []
n = 1
final_data = []
width = 160
length = 120
colour = True
num_frames = 10

outputs_path = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_datafull.npy'
train_data = np.load(outputs_path)
df = pd.DataFrame(train_data)
for data in train_data:
	image_path = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/images/image ({})'.format(n) +'.jpg'
	n = n+1
	#print(n)
	img = cv2.imread(image_path)
	#cv2.imshow('screen', img)
	#cv2.waitKey()
	#time.sleep(100)
	#image operations
	img = cv2.resize(img,(width,length),3)
	##
	if(n%1000==0):
		print(n)

	if(not(colour)):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		

	choice = list(data)
	#f.write(str(choice))
	#f.write('\r\n')
	final_data.append([img,choice])
	training_data = final_data[:12000]
	test_data = final_data[12000:14000]

np.save('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/training_rcnn_colour{0}.npy'.format(colour), training_data)
np.save('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/test_rcnn_colour{0}.npy'.format(colour), test_data)
print("FINAL DATA CREATED \nNow creating sequences")