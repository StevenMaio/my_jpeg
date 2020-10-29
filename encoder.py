'''
Does the stuff
'''

# My stuff
from utils import DCT, inverse_DCT, quantize, dequantize, RGB_to_YCbCr, YCbCr_to_RGB
from constants import WEIGHTS_DEFAULT, WEIGHTS_SMALL, QUANT_DEFAULT, DIMENSIONS

import numpy as np
from PIL import Image
from itertools import product, zip_longest

def encode(fp, output, weights=WEIGHTS_DEFAULT):
    '''
    Encodes the file whose name is given by fp into the appropriate format.
    The image must be in RGB format.

    Params:
        fp:
            an image in RGB format
        output:
            the path of the output file
    '''
    im = Image.open(fp)
    with open(output, "wb") as outfile:
        # the image will be truncated so that it's dimensions are multiples
        # of DIMENSIONS
        width_steps = im.width//DIMENSIONS
        height_steps = im.height//DIMENSIONS
        outfile.write(width_steps.to_bytes(1, 'little'))
        outfile.write(height_steps.to_bytes(1, 'little'))

        # matrices for converted colors (these can stay)
        y_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
        cb_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
        cr_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
        # iterate through each 8x8 of the image
        for step_i,step_j in product(range(width_steps), range(height_steps)):
            row_base = DIMENSIONS*step_i
            col_base = DIMENSIONS*step_j
            # convert the pixels to YCbCr format
            for i,j in product(range(DIMENSIONS),repeat=2):
                y,cb,cr = RGB_to_YCbCr(im.getpixel((row_base+i,col_base+j)))
                y_matrix[i,j] = y
                cb_matrix[i,j] = cb
                cr_matrix[i,j] = cr
            # calculate DCT of the YCrCb matrices
            transformed_y = DCT(y_matrix)
            transformed_cb = DCT(cb_matrix)
            transformed_cr = DCT(cr_matrix)
            # perform quantization
            transformed_y = quantize(transformed_y, QUANT_DEFAULT)
            transformed_cb = quantize(transformed_cb, QUANT_DEFAULT)
            transformed_cr = quantize(transformed_cr, QUANT_DEFAULT)
            indices = filter(lambda x: weights[x] != 0,
                             product(range(DIMENSIONS),repeat=2))
            for i,j in indices:
                y = int(transformed_y[i,j]).to_bytes(1,'little',signed=True)
                cb = int(transformed_cb[i,j]).to_bytes(1,'little',signed=True)
                cr = int(transformed_cr[i,j]).to_bytes(1,'little',signed=True)
                outfile.write(y)
                outfile.write(cb)
                outfile.write(cr)


def decode(fp, output, weights=WEIGHTS_DEFAULT):
    '''
    Encodes the file whose name is given by fp into the appropriate format.
    The image must be in RGB format.

    Params:
        fp:
            an image in RGB format
        output:
            the path of the output file
    '''
    with open(fp, "rb") as data:
        # FIXME: These I/O operations can be done so much better
        width_steps = int.from_bytes(data.read(1),'little')
        height_steps = int.from_bytes(data.read(1),'little')
        im = Image.new('RGB', (width_steps*DIMENSIONS, height_steps*DIMENSIONS))

        # matrices for holding vals
        transformed_y = np.zeros((DIMENSIONS, DIMENSIONS))
        transformed_cb = np.zeros((DIMENSIONS, DIMENSIONS))
        transformed_cr = np.zeros((DIMENSIONS, DIMENSIONS))
        # iterate through 8x8 block
        for step_i,step_j in product(range(width_steps),range(height_steps)):
            col_base = DIMENSIONS*step_i
            row_base = DIMENSIONS*step_j
            indices = filter(lambda x: weights[x] != 0,
                             product(range(DIMENSIONS),repeat=2))
            for pos in indices:
                # FIXME: These I/O operations can be done so much better
                y = int.from_bytes(data.read(1),'little',signed=True)
                cb = int.from_bytes(data.read(1),'little',signed=True)
                cr = int.from_bytes(data.read(1),'little',signed=True)
                transformed_y[pos] = y
                transformed_cb[pos] = cb
                transformed_cr[pos] = cr
            # convert back to RGB and then do shit
            recovered_y = dequantize(transformed_y, QUANT_DEFAULT)
            recovered_cb = dequantize(transformed_cb, QUANT_DEFAULT)
            recovered_cr = dequantize(transformed_cr, QUANT_DEFAULT)
            # do inverse DCT
            recovered_y = inverse_DCT(recovered_y)
            recovered_cb = inverse_DCT(recovered_cb)
            recovered_cr = inverse_DCT(recovered_cr)
            for i,j in product(range(DIMENSIONS),repeat=2):
                y = recovered_y[i,j]
                cb = recovered_cb[i,j]
                cr = recovered_cr[i,j]
                rgb = tuple(map(lambda t : int(t), YCbCr_to_RGB((y,cb,cr))))
                pos = (col_base+i, row_base+j)
                im.putpixel(pos, rgb)
        im.save(output)


if __name__ == '__main__':
    encode('images/test4.bmp', 'images/test4.myjeg', weights=WEIGHTS_DEFAULT)
    print("Finished encoding")
    decode('images/test4.myjeg', 'images/hi2.bmp', weights=WEIGHTS_DEFAULT)
    print("Finished decoding")
