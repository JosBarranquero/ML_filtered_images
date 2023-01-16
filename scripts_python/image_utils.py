import cv2 as cv
import numpy as np
import pandas as pd
import math

## Start of image filters section
def lowPassFilter(in_file: str, out_file: str, size: int = 3, type: int = 1):
    """This function applies a low pass filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Creating the matrix filter (sizeXsize)
    filter = np.ones((size, size))
    if (type == 1):     # Type 1 filter
        filter = 1/size**2 * filter
    elif (type == 2):   # Type 2 filter
        filter = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    elif (type == 3):   # Type 3 filter
        filter = np.zeros((size, size))
        filter[math.ceil(size/2), math.ceil(size/2)] = 1.0

    # Applying the filter
    # Second parameter = -1 : keeps the same colordepth
    result = cv.filter2D(original, -1, filter)

    # Save it to disk
    cv.imwrite(out_file, result)

def highPassFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a high pass filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Creating the matrix filter (sizeXsize)
    filter = -1/size**2 * np.ones((size, size))
    filter[math.ceil(size/2), math.ceil(size/2)] += 1.0

    # Applying the filter
    # Second parameter = -1 : keeps the same colordepth
    result = cv.filter2D(original, -1, filter)

    # Save it to disk
    cv.imwrite(out_file, result)

def medianFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a median filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)
    
    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Applying mediang filter (sizeXsize)
    result = cv.medianBlur(original, size)

    # Save it to disk
    cv.imwrite(out_file, result)

def hSobelFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a horizontal Sobel filter to a grayscale image. The resulting image is then saved to sisk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Applying horizontal Sobel filter (sizeXsize)
    result = cv.Sobel(original, -1, 1, 0, ksize=size)

    # Save it to disk
    cv.imwrite(out_file, result)

def vSobelFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a vertical Sobel filter to a grayscale image. The resulting image is then saved to sisk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Applying vertical Sobel filter (sizeXsize)
    result = cv.Sobel(original, -1, 0, 1, ksize=size)

    # Save it to disk
    cv.imwrite(out_file, result)
## End of image filters section

## Start of subimage section
def getOriginalImgSubmatrices(in_img: cv.Mat) -> pd.DataFrame:
    """Return the original image sliced intro 3x3 submatrices necessary for later processing"""
    # shape[1] is the image height, shape[0] is the image width
    height = in_img.shape[1]
    width = in_img.shape[0]
    # Submatrices will be 3x3 in size
    sub_height = 3
    sub_width = sub_height
    # List which will store the submatrices
    sub_list = []

    # To create the submatrices, the default OpenCV BorderType behavior will be replicated
    # OpenCV BORDER_REFLECT_101 reflects the pixels in the following manner gfedcb|abcdefgh|gfedcba
    # Hopefully, someday this code won't be so rough
    for i in range(0, height):  # loop through rows 
        for j in range(0, width):   # loop through columns
            cur_sub = []
            if (i == 0):    # first row
                p=0
            elif (i == (height-1)): # last row
                p=0
            else:   # rest of rows
                if (j == 0):    # first column
                    cur_sub = np.array([in_img[i-1, j+1], in_img[i-1, j], in_img[i-1, j+1], in_img[i, j+1], in_img[i, j], in_img[i, j+1], in_img[i+1, j+1], in_img[i+1, j], in_img[i+1, j+1]])
                elif (j == (height-1)): # last column
                    cur_sub = np.array([in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j-1], in_img[i, j-1], in_img[i, j], in_img[i, j-1], in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j-1]])
                else:   # rest of columns
                    cur_sub = np.array([in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j+1], in_img[i, j-1], in_img[i, j], in_img[i, j+1], in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j+1]])
    return None # TODO return something
## End of subimage section

# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")