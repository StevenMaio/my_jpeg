'''
Contains implementations of certain algorithms.
'''

import numpy as np
from itertools import product

from math import cos,pi,sqrt

from constants import weights_matrix, quant_matrix, DIMENSIONS, ONES

def _helper_alpha(t):
    if t == 0:
        return 1/sqrt(2)
    else:
        return 1

def _cos_2d(x,u):
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
    output = np.zeros((DIMENSIONS, DIMENSIONS))
    cos_arr = np.zeros((DIMENSIONS, DIMENSIONS))
    # precompute terms to save runtime
    for x,u in product(range(DIMENSIONS),repeat=2):
        cos_arr[x,u] = _cos_2d(x,u)
    # perform 2-d DCT
    for u,v in product(range(DIMENSIONS),repeat=2):
        G_uv = 0
        for x,y in product(range(DIMENSIONS),repeat=2):
            term = data[x,y]
            term *= cos_arr[x,u]*cos_arr[y,v]
            G_uv += term
        G_uv *= _helper_alpha(u)*_helper_alpha(v)/4.0
        output[u,v] = G_uv
    return output

def quantize(data, quantization_matrix):
    '''
    Performs quantization step.
    '''
    result = np.zeros((DIMENSIONS, DIMENSIONS))
    for i,j in product(range(DIMENSIONS),repeat=2):
        result[i,j] = round(data[i,j]/quantization_matrix[i,j])
    return result

def dequantize(data, quantization_matrix):
    '''
    "reverses" the quantization step. This can't be completely inverted because
    information is lossed, but it can reconstruct a close approximation.
    '''
    return np.multiply(data, quantization_matrix)

def inverse_DCT(data, weights=ONES):
    '''
    Performs the inverse 2-d DCT.

    Params:
        data:
            an array containing the DCT coefficients of some input
    Output:
        the reconstructed 8x8 matrix
    '''
    coeffs = np.multiply(data, weights) # calculate weighted DCT matrix
    output = np.zeros((DIMENSIONS, DIMENSIONS))
    cos_arr = np.zeros((DIMENSIONS, DIMENSIONS))
    # precompute terms to save runtime
    for x,u in product(range(DIMENSIONS),repeat=2):
        cos_arr[x,u] = _cos_2d(x,u)
    # calculate inverse 2-d DCT
    for x,y in product(range(DIMENSIONS),repeat=2):
        f_xy = 0
        for u,v in product(range(DIMENSIONS),repeat=2):
            term = coeffs[u,v]
            term *= cos_arr[x,u]*cos_arr[y,v]
            term *= _helper_alpha(u)*_helper_alpha(v)
            f_xy += term
        f_xy /= 4
        output[x,y] = f_xy
    return output

def RGB_to_YCbCr(p):
    '''
    Converts a pixel p from RBG to YCbCr. The particular choice of constants
    are chosen from the wikipedia article on JPEG.

    Params:
        p:
            a pixel of the form (r,g,b)
    Output:
        a pixel in YCbCr format (y,c_b,c_r)
    '''
    r,g,b = p
    y = 0.299*r + 0.587*g + 0.114*b
    cb = 128 - 0.168736*r - 0.331264*g + 0.5*b
    cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
    return (y,cb,cr)

def YCbCr_to_RGB(p):
    '''
    Converts a pixel p from YCbCr to RHB. The particular choice of constants
    are chosen from the wikipedia article on JPEG.

    Params:
        p:
            a pixel of the form (y,c_b,c_r)
    Output:
        a pixel in RGB format (r,g,b)
    '''
    y, cb, cr = p
    r = y + 1.402*(cr-128)
    g = y - 0.344136*(cb-128) - 0.714136*(cr-128)
    b = y + 1.772*(cb-128)
    return (r,g,b)


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
    g = np.matrix(m)
    out = DCT(g)
    outout = inverse_DCT(out)
    outoutout = quantize(out, quant_matrix)
    print(out)
    print(dequantize(outoutout,quant_matrix))
