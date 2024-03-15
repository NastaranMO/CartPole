import numpy as np
from scipy.signal import savgol_filter

def argmax(x):
    """Own variant of np.argmax with random tie breaking"""
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)
    
def smooth(y, window, poly=2):
    """
    y: vector to be smoothed
    window: size of the smoothing window"""
    # print('Smoothing with window size: {} and y: {}'.format(window, y))
    return savgol_filter(y, window, poly)

def softmax(x, temp):
    """Computes the softmax of vector x with temperature parameter 'temp'"""
    x = x / temp  # scale by temperature
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.exp(z) / np.sum(np.exp(z))  # compute softmax