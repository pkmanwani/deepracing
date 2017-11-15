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

width = 160
length = 120
colour = True

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
	if choice == [1,0,0,0,0,0,0,0,0]:
		keyw.append([img,choice])
	elif choice == [0,1,0,0,0,0,0,0,0]:
		keys.append([img,choice])
	elif choice == [0,0,1,0,0,0,0,0,0]:
		keya.append([img,choice])
	elif choice == [0,0,0,1,0,0,0,0,0]:
		keyd.append([img,choice])
	elif choice == [0,0,0,0,1,0,0,0,0]:
		keywa.append([img,choice])
	elif choice == [0,0,0,0,0,1,0,0,0]:
		keywd.append([img,choice])
	elif choice == [0,0,0,0,0,0,1,0,0]:
		keysa.append([img,choice])
	elif choice == [0,0,0,0,0,0,0,1,0]:
		keysd.append([img,choice])
	elif choice == [0,0,0,0,0,0,0,0,1]:
		keynk.append([img,choice])

print(len(keyw))
print(len(keys))
print(len(keya))
print(len(keyd))
print(len(keywa))
print(len(keywd))
print(len(keysa))
print(len(keysd))
print(len(keynk))
shuffle(keyw)
keyw = keyw[:3500]
final_data = []

final_data = keyw + keys + keya + keyd + keywa + keywd + keysa + keysd + keynk
print(np.shape(final_data))
shuffle(final_data)
np.save('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/training_datafull_colour{0}.npy'.format(colour), final_data)