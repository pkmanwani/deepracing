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
w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    #time.sleep(0.05)

def left():
    '''if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)'''
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(S)
    #time.sleep(0.02)

def right():
    '''if random.randrange(0,3) == 1:
    	PressKey(W)
    else:
    	ReleaseKey(W)'''
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    time.sleep(0.02)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
   #time.sleep(0.01)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    #time.sleep(0.01)
    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    ''''if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)'''
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def releaseall():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)

cnnmodel = load_model('saved_models/weights_best_new.h5')
print("Loaded cnn model")

#rnnmodel=load_model('saved_models/rnn_weights_best.h5')
#print("Loaded rnn model")

'''for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
'''
while(True):
    last_time = time.time()
    screen = grab_screen(region = (0,23, 800, 623))
    screen = screen[:,:,:3]
    #cv2.imwrite('image.jpg', screen)
    #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(width,length),3)
    inputs = screen.flatten()
    inputs  = screen.reshape(1,length,width,3)
    inputs = inputs.astype('float32')
    rough_out = cnnmodel.predict(inputs)[0]
    prediction = np.round(rough_out)
    mode_choice = np.argmax(prediction)
    #releaseall()
    if mode_choice == 0:
        straight()
        choice_picked = 'straight'
    if mode_choice == 1:
        reverse()
        choice_picked = 'reverse'
    elif mode_choice == 2:
        left()
        choice_picked = 'left'
    elif mode_choice == 3:
        right()
        choice_picked = 'right'
    elif mode_choice == 4:
        forward_left()
        choice_picked = 'forward+left'
    elif mode_choice == 5:
        forward_right()
        choice_picked = 'forward+right'
    elif mode_choice == 6:
        reverse_left()
        choice_picked = 'reverse+left'
    elif mode_choice == 7:
        reverse_right()
        choice_picked = 'reverse+right'
    elif mode_choice == 8:
        no_keys()
        choice_picked = 'nokeys'
    print(choice_picked)
    #time.sleep(0.3)
    #releaseall()
    #ReleaseKey(W)
    #ReleaseKey(A)
    #ReleaseKey(S)
    #ReleaseKey(D)
    
    '''if(last_choice == choice_picked):
    print('\n')
    else:
        print(choice_picked)
    last_choice = choice_picked'''
    #print(last_time-time.time())