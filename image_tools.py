'''
Tools for processing images, i.e., color space transformations, converting
images into matrices, etc.

TODO:
    - these functions use constants from the wikipedia article. It will
      probably be nice to revisit these functions and make them more flexible
'''

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
