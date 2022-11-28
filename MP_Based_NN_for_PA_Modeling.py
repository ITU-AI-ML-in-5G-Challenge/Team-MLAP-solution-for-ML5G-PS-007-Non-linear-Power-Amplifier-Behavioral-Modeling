
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, LeakyReLU
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Adam

import math
import scipy.signal as signal

'''
Defining variable parameters
'''

max_order = 2 # Max mag power fed at input
memLength = 11 # Memory Length
seqLength = memLength+1 # previous memLength samples and current sample

HL1C = 9 # Number of nodes in hidden layer 1 for current sample
HL2C = 5 # Number of nodes in hidden layer 2 for current sample
HL1P = 5 # Number of nodes in hidden layer 1 for past samples
HL2P = 2 # Number of nodes in hidden layer 2 for past samples
FC1 = 10 # Number of nodes in hidden layer 3
FC2 = 20 # Number of nodes in hidden layer 4
FC3 = 10 # Number of nodes in hidden layer 5

aph1 = 0.5
aph2 = 0.2

bs = 200

train_enable = 0 # 0 for testing and 1 for training


Total_NMSE = 0
Total_ACEPR = 0

'''
Load Data
'''

if train_enable == 1:
    dataset = ['DataTrain']
elif train_enable == 0:
    dataset = ['DataTrain', 'DataTest1', 'DataTest2', 'DataTest3','DataTest4']

for dataStr in dataset:
    folder_name = './Datasets/' # Path to folder containing datasets
    data = loadmat(folder_name+dataStr+'.mat')

    inpPA_whole = data['in']
    outPA_whole = data['out']

    #Disable the commented part in the below 2 lines to find the normalizing factor if you are training with new dataset

    '''
    pos = np.argmax(abs(inpPA_whole))
    normalizing_factor = inpPA_whole[pos]

    '''
    normalizing_factor = [abs(1690.16376092-11735.22170986j)] # Identified the max of absolute value in the input for the given training set

    '''
    Normalizing both input and output
    '''

    inpPA = inpPA_whole/normalizing_factor 
    outPA = outPA_whole/normalizing_factor

    [no_samples,_] = np.shape(inpPA)

    '''
    Data processing to form input vectors and their corresponding output vectors
    '''

    inpPA_complete = (max_order+2)*[1]

    inpPA_complete[0] = np.real(inpPA)
    inpPA_complete[1] = np.imag(inpPA)

    for v in range(max_order):
        inpPA_complete[v+2] = np.power(np.abs(inpPA), (v+1))


    outPA_complete = (max_order+2)*[1]

    outPA_complete[0] = np.real(outPA)
    outPA_complete[1] = np.imag(outPA)

    for v in range(max_order):
        outPA_complete[v+2] = np.power(np.abs(outPA), (v+1))
        

    inputVec_list = seqLength*[1]
    outputVec_new = np.zeros((no_samples-seqLength+1,2))

    for v in range(seqLength):
        inputVec_list[v] = np.zeros((no_samples-seqLength+1,(max_order+2)))

    for i in range (no_samples-seqLength+1):
        for j in range (seqLength):
            
            inputVec_list[j][i,0] = inpPA_complete[0][i+j]
            inputVec_list[j][i,1] = inpPA_complete[1][i+j]
            for k in range(max_order):
                inputVec_list[j][i,k+2] = inpPA_complete[k+2][i+j]

        outputVec_new[i,0] = outPA_complete[0][i+seqLength-1]
        outputVec_new[i,1] = outPA_complete[1][i+seqLength-1]




    '''
    Defining Network Architecture
    '''

    input_shape = (inputVec_list[0].shape[1], )

    input_layer = seqLength*[1]
    dense_layer1_s = (seqLength)*[1]
    dense_layer1_s2 = (seqLength)*[1]
    dla1 = (seqLength)*[1]
    dla2 = (seqLength)*[1]

    for k in range(seqLength):
        input_layer[k] = Input(shape=input_shape)

        if k==seqLength-1: # Current sample
            dense_layer1_s[k] = Dense((HL1C), activation='linear')(input_layer[k])
            dla1[k] = LeakyReLU(alpha=aph1)(dense_layer1_s[k])
            dense_layer1_s2[k] = Dense((HL2C), activation='linear')(dla1[k])
            dla2[k] = LeakyReLU(alpha=aph2)(dense_layer1_s2[k])
        else: # Past samples
            dense_layer1_s[k] = Dense((HL1P), activation='linear')(input_layer[k])
            dla1[k] = LeakyReLU(alpha=aph1)(dense_layer1_s[k])
            dense_layer1_s2[k] = Dense((HL2P), activation='linear')(dla1[k])
            dla2[k] = LeakyReLU(alpha=aph2)(dense_layer1_s2[k])

    hidden_merged_layer1 = Concatenate()(dla2)

    hidden_layer2 = Dense(FC1, activation='linear')(hidden_merged_layer1)
    dlh2a = tanh(hidden_layer2)

    HL_3 = Dense(FC2, activation='linear')(dlh2a)
    dlh3a = tanh(HL_3)

    HL_4 = Dense(FC3, activation='linear')(dlh3a)
    dlh4a = tanh(HL_4)

    final_layer = Dense(2, activation='linear')(dlh4a)


    model = Model(inputs=input_layer,
                  outputs=final_layer)

    name = 'memLength_'+str(memLength)+'.h5'

    '''
    Defining the data vectors
    '''


    X_train = inputVec_list

    Y_train = outputVec_new

    '''
    Training to fit the parameters
    '''
    
    
    if train_enable == 1:

        for train_number in range(2):
            if train_number == 0:
                LR = 1e-3
                ep = 3000
            elif train_number == 1:
                LR = 1e-5
                ep = 300
                model.load_weights(name)

            model.compile(optimizer=Adam(learning_rate=LR), loss='mean_squared_error', metrics=['mean_squared_error'])

            model.fit(X_train, Y_train, epochs=ep, batch_size = bs, verbose=2)
            model.save_weights(name)


    if train_enable == 0:

        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mean_squared_error'])

        print('\nMetrics for dataset - '+dataStr+'\n')


        # Obtaining the metrics

        model.load_weights(name)

        a = model.predict(inputVec_list, verbose=2) # Prediction using network

        '''
        Converting back to complex number
        '''

        Y_out = np.zeros((len(a),1),dtype=np.complex128) 

        for i in range (len(a)):
            Y_out[i,0] = complex(a[i,0],a[i,1])


        Y_pred = Y_out*normalizing_factor # multiplying using normalization factor to get the actual

        out_ignore_first_mem = outPA_whole[seqLength-1:no_samples]# First M samples are neglected
        inp_ignore_first_mem = inpPA_whole[seqLength-1:no_samples]
        error = Y_pred-out_ignore_first_mem

        '''
        NMSE Calculation
        '''
        def NMSE(error, out_ignore_first_mem):
                
            num = np.mean(np.square(np.abs(error)))
            den = np.mean(np.square(np.abs(out_ignore_first_mem)))

            nmse = 10*math.log((num/den), 10)
            
            print('NMSE = ', nmse)

        '''
        ACEPR Calculation
        '''
        def ACEPR_cal(error, inp_ignore_first_mem):
            Z = error + inp_ignore_first_mem
            Z= Z.reshape(len(Z),)

            fs = 983.04e6
            nfft = len(Z)
            window = signal.hann(len(Z), False)

            freq, Pin = signal.periodogram(Z, fs, window, nfft)

            Pin = np.fft.fftshift(Pin)
            Freq = np.fft.fftshift(freq)

            P0 = 0
            P1 = 0
            P2 = 0

            for i in range(len(Freq)):
                if Freq[i] > - 100e6 and Freq[i] < 100e6:
                    P0 += Pin[i]

                elif Freq[i] > - 200e6 and Freq[i] < -100e6:
                    P1 += Pin[i]

                elif Freq[i] > 100e6 and Freq[i] < 200e6:
                    P2 += Pin[i]

            ACEPR_left = 10*math.log((P1/P0), 10)
            ACEPR_right = 10*math.log((P2/P0), 10)
            ACEPR = 0.5*(ACEPR_left+ACEPR_right)

            print('ACEPR = ', ACEPR)

        NMSE(error, out_ignore_first_mem)
        ACEPR_cal(error, inp_ignore_first_mem)

        if dataStr == 'DataTest1' or dataStr == 'DataTest2':

            piece1_end = 20000

            piece1_pred = Y_pred[0:piece1_end-seqLength+1]
            piece1_out = outPA_whole[seqLength-1:piece1_end]

            piece1_in = inpPA_whole[seqLength-1:piece1_end]
            error = piece1_pred-piece1_out

            print('For piece1')

            NMSE(error, piece1_out)
            ACEPR_cal(error, piece1_in)

            
            piece2_pred =  Y_pred[piece1_end:]
            piece2_out = outPA_whole[piece1_end+seqLength-1:]

            piece2_in = inpPA_whole[piece1_end+seqLength-1:]
            error = piece2_pred-piece2_out

            print('For piece2')

            NMSE(error, piece2_out)
            ACEPR_cal(error, piece2_in)
