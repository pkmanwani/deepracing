import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os
keyw = []
keys = []
keya = []
keyd = []
keywa = []
keywd = []
keysa = []
keysd = []
keynk= []


for i in range(11,33):
	train_data = np.load('test_data-{}.npy'.format(i))
	df = pd.DataFrame(train_data)
	for data in train_data:
		img = data[0]
		choice = data[1]
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
keyw = keyw[:5000]
print(np.shape(keyw))
old_data = []
old_data = np.load('training_finaldata2.npy')
print(np.shape(old_data))
final_data = keyw + keys + keya + keyd + keywa + keywd + keysa + keysd + keynk
for data in old_data:
		img = data[0]
		choice = data[1]
		final_data.append([img,choice])
print(np.shape(final_data))
shuffle(final_data)
np.save('training_finaldata3.npy', final_data)