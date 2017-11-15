import numpy as np
total_data = np.load('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_data.npy')
train_data = np.load('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_datanew.npy')
total_data = np.vstack((train_data, total_data))
training_data = np.save('C:/Users/pratik pc/Desktop/cnnproj/data/training/track1/outputs/training_datafull.npy', total_data)