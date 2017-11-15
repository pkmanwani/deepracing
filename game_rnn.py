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
    #time.sleep(0.05)

def right():
    '''if random.randrange(0,3) == 1:
    	PressKey(W)
    else:
    	ReleaseKey(W)'''
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    #time.sleep(0.05)
    
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

def add_lists(a, b):
	for i,val in enumerate(a):
		a[i] = a[i] + b[i]
	return a

def mul_lists(a, b):
    for i,val in enumerate(a):
        a[i] = a[i]*b[i]
    return a

rnnmodel = load_model('saved_models/rnn_weights_best2.h5')
print("Loaded model")

cnn_middle = load_model('saved_models/weights_best_new.h5')
print("Loaded cnn model")

cnn_middle.pop()
cnn_middle.pop()
cnn_middle.pop()

cnnmodel = load_model('saved_models/weights_best_new.h5')
print("Loaded cnn model")


rnn_in = []
flag = 0
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
    cnn_out = cnn_middle.predict(inputs)[0]
    cnn_pred = cnnmodel.predict(inputs)[0]
    if(len(rnn_in)==6):
        flag=1
        rnn_in.append(cnn_out)
        #print(np.shape(rnn_in))
        rnn_in = rnn_in[1:7]
        #print(np.shape(rnn_in))
        rnn_input = np.reshape(rnn_in, (1, 6, 512))
        #print(np.shape(rnn_input))
        rnn_out = rnnmodel.predict(rnn_input)[0]
        #print(rnn_out)
        #prediction = np.round(list(rnn_out))
        #print(prediction)
        #a = [0.4, 0, 0, 0 ,0,0, 0, 0, 0]
        #b = list(rnn_out)
        #print(list(rnn_out))
        #print(list(cnn_pred))
        #c = substract_lists(b,a)
        #print(list(rnn_out))
        #print()
        rnn_choice = np.argmax(list(rnn_out))
        cnn_choice = np.argmax(list(cnn_pred))
    	#print(mode_choice)
        rnn_mult= [0.2,1,1,1,1.1,1,1,1,1] 
        cnn_mult = [2,1,1,1,1,1,1,1,1]
        x = list(rnn_out)
        y = list(cnn_pred)
        #print(x)
        #print(y)
        c = mul_lists(rnn_mult,x)
        d = mul_lists(cnn_mult,y)
        e = add_lists(c,d)
        #print(c,d)
        mode_choice = np.argmax(e)
    else:
        rnn_in.append(cnn_out)
        #print(np.shape(rnn_in))
        mode_choice = 8
        cnn_choice = 8
    print(mode_choice)
    print(cnn_choice)
    #mode_choice = cnn_choice
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
    time.sleep(0.3)
    releaseall()
    '''if(last_choice == choice_picked):
    print('\n')
    else:
        print(choice_picked)
    last_choice = choice_picked'''
    #print(last_time-time.time())