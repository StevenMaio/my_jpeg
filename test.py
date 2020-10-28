'''
A sample attempt at this compression algorithm.
'''

# My stuff
from algorithms import DCT, inverse_DCT, quantize, dequantize
from utils import weights_matrix, quant_matrix, DIMENSIONS
from image_tools import RGB_to_YCbCr, YCbCr_to_RGB

# Other stuff
import numpy as np
from PIL import Image
from itertools import product

def checkerboard_test():
    # open image
    image_path = "images/8x8_board.png"
    output_path = "images/8x8_board_foodled.png"
    im = Image.open(image_path)

    # matrices for converted colors
    y_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
    cr_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
    cb_matrix = np.zeros((DIMENSIONS,DIMENSIONS))

    # calculate YCrCb representation
    for i,j in product(range(DIMENSIONS),repeat=2):
        y,cb,cr = RGB_to_YCbCr(im.getpixel((i,j)))
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

    # dequantize matrices
    recovered_y = dequantize(transformed_y, quant_matrix)
    recovered_cb = dequantize(transformed_cb, quant_matrix)
    recovered_cr = dequantize(transformed_cr, quant_matrix)

    # inverse DCT while ignoring bottom right quadrant
    recovered_y = inverse_DCT(recovered_y, weights_matrix)
    recovered_cb = inverse_DCT(recovered_cb, weights_matrix)
    recovered_cr = inverse_DCT(recovered_cr, weights_matrix)

    # reconstruct the original image
    reconstructed_im = Image.new('RGB', (DIMENSIONS, DIMENSIONS))
    for i,j in product(range(DIMENSIONS),repeat=2):
        # convert each pixel from YCbCr to RGB
        y = recovered_y[i,j]
        cb = recovered_cb[i,j]
        cr = recovered_cr[i,j]
        p = YCbCr_to_RGB((y,cb,cr))
        r,g,b = p
        r = int(r)
        g = int(g)
        b = int(b)
        reconstructed_im.putpixel((i,j), (r,g,b))

    # save reconstructed image
    reconstructed_im.save(output_path)


if __name__ == '__main__':
    checkerboard_test()
