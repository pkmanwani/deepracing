import numpy as np


input_data = []
#name of file that contains reduced data
file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/training_rcnn_colourTrue.npy'
input_trainingdata = np.load(file_name1)
file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/test_rcnn_colourTrue.npy'
input_testdata = np.load(file_name2)
#split training and test data into x and y
x_train = []
y_train = []
count = 0
for data in input_trainingdata:
	img = data[0]
	#print(img)
	choice = data[1]
	length = img.shape[0]
	width = img.shape[1]	
	flat_img = img.flatten()
	#print(flat_img)
	if(count==0):
		x_train = flat_img
		y_train = choice
	else:
		x_train = np.vstack((x_train, flat_img))
		y_train = np.vstack((y_train, choice))
	count = count +1
count = 0
x_test = []
y_test = []
for data in input_testdata:
	img = data[0]
	choice = data[1]
	flat_img = img.flatten()
	if(count==0):
		x_test = flat_img
		y_test = choice
	else:
		x_test = np.vstack((x_test, flat_img))
		y_test = np.vstack((y_test, choice))
	count = count +1
#now we have x_train, x_test, y_train, y_test

#x_train = x_train.reshape(x_train.shape[0],width ,length, 1)
#x_test = x_test.reshape(x_test.shape[0], width, length, 1)

file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/xtrain_data.npy'
np.save(file_name1, x_train)
file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/xtest_data.npy'
np.save(file_name2, x_test)
file_name1 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/ytrain_data.npy'
np.save(file_name1, y_train)
file_name2 = 'C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/ytest_data.npy'
np.save(file_name2, y_test)

