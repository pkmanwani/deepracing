import numpy as np
import os
import sys
fwrite = open('choices.txt', 'w')

x= 0
for i in range(1,29):
	#print(i)
	train_data = np.load('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_data ({}).npy'.format(i))
	for data in train_data:
		x = x+1
		choice = list(data)
		if choice == [1,0,0,0,0,0,0,0,0]:
			#keyw.append([choice])
			a = 'forward'
		elif choice == [0,1,0,0,0,0,0,0,0]:
			#keys.append([choice])
			a = 'reverse'
		elif choice == [0,0,1,0,0,0,0,0,0]:
			#keya.append([choice])
			a = 'left'
		elif choice == [0,0,0,1,0,0,0,0,0]:
			#keyd.append([choice])
			a = 'right'
		elif choice == [0,0,0,0,1,0,0,0,0]:
			#keywa.append([choice])
			a = 'forward+left'
		elif choice == [0,0,0,0,0,1,0,0,0]:
			#keywd.append([choice])
			a = 'forward+right'
		elif choice == [0,0,0,0,0,0,1,0,0]:
			#keysa.append([choice])
			a = 'reverse+left'
		elif choice == [0,0,0,0,0,0,0,1,0]:
			#keysd.append([choice])
			a = 'reverse+right'
		elif choice == [0,0,0,0,0,0,0,0,1]:
			#keynk.append([choice])
			choice = 'nokeys'
		fwrite.write(str(x) + a+'\n')
fwrite.close()
x= 0
fwrite = open('choices1.txt', 'w')
train_data = np.load('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_datafull.npy')
for data in train_data:
		x = x+1
		choice = list(data)
		if choice == [1,0,0,0,0,0,0,0,0]:
			#keyw.append([choice])
			a = 'forward'
		elif choice == [0,1,0,0,0,0,0,0,0]:
			#keys.append([choice])
			a = 'reverse'
		elif choice == [0,0,1,0,0,0,0,0,0]:
			#keya.append([choice])
			a = 'left'
		elif choice == [0,0,0,1,0,0,0,0,0]:
			#keyd.append([choice])
			a = 'right'
		elif choice == [0,0,0,0,1,0,0,0,0]:
			#keywa.append([choice])
			a = 'forward+left'
		elif choice == [0,0,0,0,0,1,0,0,0]:
			#keywd.append([choice])
			a = 'forward+right'
		elif choice == [0,0,0,0,0,0,1,0,0]:
			#keysa.append([choice])
			a = 'reverse+left'
		elif choice == [0,0,0,0,0,0,0,1,0]:
			#keysd.append([choice])
			a = 'reverse+right'
		elif choice == [0,0,0,0,0,0,0,0,1]:
			#keynk.append([choice])
			choice = 'nokeys'
		fwrite.write(str(x) + a+'\n')