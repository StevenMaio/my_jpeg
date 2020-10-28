'''
Contains implementations of certain algorithms.
'''

import numpy as np
from itertools import product

from math import cos,pi,sqrt

NUM_ROWS = 8
NUM_COLS = 8

def helper_alpha(t):
    if t == 0:
        return 1/sqrt(2)
    else:
        return 1

def cos_2d(x,u):
    return cos((2*x+1)*u*pi/16)

def DCT(data):
    '''
    Performs the 2-d DCT algorithm on matrix data. For now, we'll assume the
    data is an 8x8 array.

    Params:
        data:
            an array of data
    Output:
        the DCT of the input data
    '''
    output = np.zeros((NUM_ROWS, NUM_COLS))
    cos_arr = np.zeros((NUM_ROWS, NUM_COLS))
    # precompute terms to save runtime
    for x,u in product(range(8),repeat=2):
        cos_arr[x,u] = cos_2d(x,u)
    # perform 2-d DCT
    for u,v in product(range(8),repeat=2):
        G_uv = 0
        for x,y in product(range(8),repeat=2):
            term = data[x,y]
            term *= cos_arr[x,u]
            term *= cos_arr[y,v]
            G_uv += term
        G_uv *= helper_alpha(u)*helper_alpha(v)/4.0
        output[u,v] = G_uv
    return output

def inverse_DCT(data, weights=np.ones((NUM_ROWS, NUM_ROWS))):
    '''
    Performs the inverse 2-d DCT.

    Params:
        data:
            an array containing the DCT coefficients of some input
    Output:
        the reconstructed 8x8 matrix
    '''
    coeffs = np.multiply(data, weights) # calculate weighted DCT matrix
    output = np.zeros((NUM_ROWS, NUM_COLS))
    cos_arr = np.zeros((NUM_ROWS, NUM_COLS))
    # precompute terms to save runtime
    for x,u in product(range(8),repeat=2):
        cos_arr[x,u] = cos_2d(x,u)
    # calculate inverse 2-d DCT
    for x,y in product(range(8),repeat=2):
        f_xy = 0
        for u,v in product(range(8),repeat=2):
            term = coeffs[u,v]
            term *= cos_arr[x,u]
            term *= cos_arr[y,v]
            term *= helper_alpha(u)*helper_alpha(v)
            f_xy += term
        f_xy /= 4
        output[x,y] = f_xy
    return output

if __name__ == '__main__':
    m = [
            [-76,-73,-67,-62,-58,-67,-64,-55],
            [-65,-69,-73,-38,-19,-43,-59,-56],
            [-66,-69,-60,-15,16,-24,-62,-55],
            [-65,-70,-57,-6,26,-22,-58,-59],
            [-61,-67,-60,-24,-2,-40,-60,-58],
            [-49,-63,-68,-58,-51,-60,-70,-53],
            [-43,-57,-64,-69,-73,-67,-63,-45],
            [-41,-49,-59,-60,-63,-52,-50,-34]
    ]
    my_weights = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0,0],
    ]
    g = np.matrix(m)
    out = DCT(g)
    outout = inverse_DCT(out, weights=my_weights)
    print(g-outout)
