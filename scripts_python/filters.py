import cv2 as cv
import numpy as np
import math

def low_pass_filter(in_file: str, out_file: str, size: int = 3):
    """This function applies a low pass filter to a grayscale image. The resulting image is then save to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_UNCHANGED)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    if original.ndim == 3:   # If it's a color image, raise an error
        raise RuntimeError('Image \'{0}\' is in color!'.format(in_file))

    # Creating the matrix filter (sizeXsize)
    filter = 1/size**2 * np.ones((size, size))     # Type 1 filter
    # filter = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])   # Type 2 filter
    # filter = np.zeros((size, size))     # Type 3 filter
    # filter[math.ceil(size/2), math.ceil(size/2)] = 1.0

    # Applying the filter
    # Second parameter = -1 : keeps the same colordepth
    result = cv.filter2D(original, -1, filter)

    # Save it to disk
    cv.imwrite(out_file, result)

def high_pass_filter(in_file: str, out_file: str, size: int = 3):
    """This function applies a high pass filter to a grayscale image. The resulting image is then save to disk"""
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

def median_filter(in_file: str, out_file: str, size: int = 3):
    """This function applies a median filter to a grayscale image. The resulting image is then save to disk"""
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

# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")