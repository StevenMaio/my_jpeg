'''
Does the stuff
'''

# My stuff
from algorithms import DCT, inverse_DCT, quantize, dequantize, RGB_to_YCbCr, YCbCr_to_RGB
from constants import weights_matrix, weights_matrix2, quant_matrix, DIMENSIONS

import numpy as np
from PIL import Image
from itertools import product, zip_longest

def encode(fp, output):
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
        for step_i,step_j in product(range(height_steps), range(width_steps)):
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
            transformed_y = quantize(transformed_y, quant_matrix)
            transformed_cb = quantize(transformed_cb, quant_matrix)
            transformed_cr = quantize(transformed_cr, quant_matrix)
            # need to figure out a good way to undo this -- or it won't be
            # a problem?
            indices = filter(lambda x: min(x[0],x[1]) <= DIMENSIONS//2,
                             product(range(DIMENSIONS),repeat=2))
            for i,j in indices:
                y = int(transformed_y[i,j]).to_bytes(1,'little',signed=True)
                cb = int(transformed_cb[i,j]).to_bytes(1,'little',signed=True)
                cr = int(transformed_cr[i,j]).to_bytes(1,'little',signed=True)
                outfile.write(y)
                outfile.write(cb)
                outfile.write(cr)


def decode(fp, output):
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
        width_steps = int.from_bytes(data.read(1),'little')
        height_steps = int.from_bytes(data.read(1),'little')
        im = Image.new('RGB', (width_steps*DIMENSIONS, height_steps*DIMENSIONS))

        # matrices for holding vals
        transformed_y = np.zeros((DIMENSIONS, DIMENSIONS))
        transformed_cb = np.zeros((DIMENSIONS, DIMENSIONS))
        transformed_cr = np.zeros((DIMENSIONS, DIMENSIONS))
        # iterate through 8x8 block
        for step_i,step_j in product(range(height_steps),range(width_steps)):
            i_base = DIMENSIONS*step_i
            j_base = DIMENSIONS*step_j
            indices = filter(lambda x: min(x[0],x[1]) <= DIMENSIONS//2,
                             product(range(DIMENSIONS),repeat=2))
            for pos in indices:
                y = int.from_bytes(data.read(1),'little',signed=True)
                cb = int.from_bytes(data.read(1),'little',signed=True)
                cr = int.from_bytes(data.read(1),'little',signed=True)
                transformed_y[pos] = y
                transformed_cb[pos] = cb
                transformed_cr[pos] = cr
            # convert back to RGB and then do shit
            recovered_y = dequantize(transformed_y, quant_matrix)
            recovered_cb = dequantize(transformed_cb, quant_matrix)
            recovered_cr = dequantize(transformed_cr, quant_matrix)
            # do inverse DCT
            recovered_y = inverse_DCT(recovered_y)
            recovered_cb = inverse_DCT(recovered_cb)
            recovered_cr = inverse_DCT(recovered_cr)
            for i,j in product(range(DIMENSIONS),repeat=2):
                y = recovered_y[i,j]
                cb = recovered_cb[i,j]
                cr = recovered_cr[i,j]
                r,g,b = YCbCr_to_RGB((y,cb,cr))
                r = int(r)
                g = int(g)
                b = int(b)
                pos = (i_base+i, j_base+j)
                val = (r,g,b)
                im.putpixel(pos, val)
        im.save(output)


if __name__ == '__main__':
    encode('images/test3.bmp', 'images/test3.myjeg')
    print("Finished encoding")
    decode('images/test3.myjeg', 'images/hi.bmp')
    print("Finished decoding")
