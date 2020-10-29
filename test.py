'''
A sample attempt at this compression algorithm.
'''

# My stuff
from utils import DCT, inverse_DCT, quantize, dequantize, RGB_to_YCbCr, YCbCr_to_RGB
from constants import WEIGHTS_DEFAULT, WEIGHTS_SMALL, quant_matrix, DIMENSIONS

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
    recovered_y = inverse_DCT(recovered_y, WEIGHTS_DEFAULT)
    recovered_cb = inverse_DCT(recovered_cb, WEIGHTS_DEFAULT)
    recovered_cr = inverse_DCT(recovered_cr, WEIGHTS_DEFAULT)

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


# second test
def test2():
    # open image
    image_path = "images/test2.bmp"
    output_path = "images/test2_foodled.bmp"
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
    recovered_y = inverse_DCT(recovered_y, WEIGHTS_DEFAULT)
    recovered_cb = inverse_DCT(recovered_cb, WEIGHTS_DEFAULT)
    recovered_cr = inverse_DCT(recovered_cr, WEIGHTS_DEFAULT)

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

# third test -- reconstructing a bigger image
def test3():
    # open image
    image_path = "images/test3.bmp"
    output_path = "images/test3_foodled2.bmp"
    im = Image.open(image_path)
    reconstructed_im = Image.new('RGB', (im.width, im.height))

    m = 128//DIMENSIONS
    for a,b in product(range(m),repeat=2):
        # base of image
        x_base = DIMENSIONS*a
        y_base = DIMENSIONS*b

        # matrices for converted colors
        y_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
        cr_matrix = np.zeros((DIMENSIONS,DIMENSIONS))
        cb_matrix = np.zeros((DIMENSIONS,DIMENSIONS))

        # calculate YCrCb representation
        for i,j in product(range(DIMENSIONS),repeat=2):
            y,cb,cr = RGB_to_YCbCr(im.getpixel((x_base+i,y_base+j)))
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
        recovered_y = inverse_DCT(recovered_y, WEIGHTS_DEFAULT)
        recovered_cb = inverse_DCT(recovered_cb, WEIGHTS_DEFAULT)
        recovered_cr = inverse_DCT(recovered_cr, WEIGHTS_DEFAULT)

        # reconstruct the original image
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
            reconstructed_im.putpixel((x_base+i,y_base+j), (r,g,b))
        print("Finished {}-{},{}-{}".format(x_base,x_base+DIMENSIONS,y_base,y_base+DIMENSIONS))

    # save reconstructed image
    reconstructed_im.save(output_path)


if __name__ == '__main__':
    test3()
