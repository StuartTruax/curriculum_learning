import numpy as np
import pandas as pd
import math


def sine_random_phase(A,f,fs,T,tt):

    random_phase = np.random.uniform(-np.pi,np.pi)
    return A*np.sin(2*np.pi*f*tt+random_phase)


def additive_gaussian_noise(waveform,mu,sigma):
    return waveform+np.random.normal(mu,sigma, size = len(waveform))


def time_window(X,Y, look_back):
    """
    Y:                  [[i]]
                          ^
                          |
         __________________
        |                   |
    X: [[i-look_back]..[i]]
    """
    X_time_windowed = []
    Y_shifted = []
    for i in range(look_back,len(X)):
        X_time_windowed.append(X[(i-look_back):i+1].flatten())
        Y_shifted.append(Y[i])
    return np.array(X_time_windowed), np.array(Y_shifted)


def scale_vector(X):
    X = X.reshape((X.shape[0],1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    X = X.flatten()
    return X

def reshape_for_keras(X, num_samples,num_features,len_sequence ):
    return X.reshape((num_samples,num_features, len_sequence))
