import cv2 as cv
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio, normalized_mutual_information
import math

## Start of image filters section
def lowPassFilter(in_file: str, out_file: str, size: int = 3, type: int = 1):
    """This function applies a low pass filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_GRAYSCALE)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    # Creating the matrix filter (sizeXsize)
    filter = np.ones((size, size))
    if (type == 1):     # Type 1 filter
        filter = 1/size**2 * filter
    elif (type == 2):   # Type 2 filter
        filter = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    elif (type == 3):   # Type 3 filter
        filter = np.array([[0, 0.125, 0], [0.125, 0.5, 0.125], [0, 0.125, 0]])

    # Applying the filter
    # Second parameter = -1 : keeps the same colordepth
    result = cv.filter2D(original, -1, filter)

    # Save it to disk
    cv.imwrite(out_file, result)

def highPassFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a high pass filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_GRAYSCALE)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    # Creating the matrix filter (sizeXsize)
    filter = -1/size**2 * np.ones((size, size))
    filter[math.floor(size/2), math.floor(size/2)] += 1.0

    # Applying the filter
    # Second parameter = -1 : keeps the same colordepth
    result = cv.filter2D(original, -1, filter)

    # Save it to disk
    cv.imwrite(out_file, result)

def medianFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a median filter to a grayscale image. The resulting image is then saved to disk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_GRAYSCALE)
    
    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    # Applying mediang filter (sizeXsize)
    result = cv.medianBlur(original, size)

    # Save it to disk
    cv.imwrite(out_file, result)

def hSobelFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a horizontal Sobel filter to a grayscale image. The resulting image is then saved to sisk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_GRAYSCALE)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    # Applying horizontal Sobel filter (sizeXsize)
    result = cv.Sobel(original, -1, 1, 0, ksize=size)

    # Save it to disk
    cv.imwrite(out_file, result)

def vSobelFilter(in_file: str, out_file: str, size: int = 3):
    """This function applies a vertical Sobel filter to a grayscale image. The resulting image is then saved to sisk"""
    # Reading the input image
    original = cv.imread(in_file, cv.IMREAD_GRAYSCALE)

    if original is None:    # If image wasn't read, then the file doesn't exist
        raise FileNotFoundError('Image \'{0}\' not found'.format(in_file))

    # Applying vertical Sobel filter (sizeXsize)
    result = cv.Sobel(original, -1, 0, 1, ksize=size)

    # Save it to disk
    cv.imwrite(out_file, result)
## End of image filters section

## Start of subimage section
def getOriginalImgSubmatrices(in_img: cv.Mat) -> pd.DataFrame:
    """Return the original image sliced intro 3x3 submatrices necessary for later processing"""
    # shape[0] is the image height, shape[1] is the image width
    (height, width) = in_img.shape
    # List which will store the submatrices
    sub_list = []

    # To create the submatrices, the default OpenCV BorderType behavior will be replicated
    # OpenCV BORDER_REFLECT_101 reflects the pixels in the following manner gfedcb|abcdefgh|gfedcba
    # Hopefully, someday this code won't be so rough
    for i in range(0, height):  # loop through rows 
        for j in range(0, width):   # loop through columns
            cur_sub = []    # current iteration submatrix
            if (i == 0):    # first row
                if (j == 0):    # first column
                    cur_sub = np.array(
                        [in_img[i+1, j+1], in_img[i+1, j], in_img[i+1, j+1],
                        in_img[i, j+1], in_img[i, j], in_img[i, j+1],
                        in_img[i+1, j+1], in_img[i+1, j], in_img[i+1, j+1]])
                elif (j == (width-1)): # last column
                    cur_sub = np.array(
                        [in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j-1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j-1],
                        in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j-1]])
                else:   # rest of columns
                    cur_sub = np.array(
                        [in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j+1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j+1],
                        in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j+1]])
            elif (i == (height-1)): # last row
                if (j == 0):    # first column
                    cur_sub = np.array(
                        [in_img[i-1, j+1], in_img[i-1, j], in_img[i-1, j+1],
                        in_img[i, j+1], in_img[i, j], in_img[i, j+1],
                        in_img[i-1, j+1], in_img[i-1, j], in_img[i-1, j+1]])
                elif (j == (width-1)): # last column
                    cur_sub = np.array(
                        [in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j-1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j-1],
                        in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j-1]])
                else:   # rest of columns
                    cur_sub = np.array(
                        [in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j+1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j+1],
                        in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j+1]])
            else:   # rest of rows
                if (j == 0):    # first column
                    cur_sub = np.array(
                        [in_img[i-1, j+1], in_img[i-1, j], in_img[i-1, j+1],
                        in_img[i, j+1], in_img[i, j], in_img[i, j+1],
                        in_img[i+1, j+1], in_img[i+1, j], in_img[i+1, j+1]])
                elif (j == (width-1)): # last column
                    cur_sub = np.array(
                        [in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j-1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j-1],
                        in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j-1]])
                else:   # rest of columns
                    cur_sub = np.array(
                        [in_img[i-1, j-1], in_img[i-1, j], in_img[i-1, j+1],
                        in_img[i, j-1], in_img[i, j], in_img[i, j+1],
                        in_img[i+1, j-1], in_img[i+1, j], in_img[i+1, j+1]])
            sub_list.append(cur_sub)

    return pd.DataFrame(np.array(sub_list, dtype=np.uint8))

def getFilteredImgSubmatrices(in_img: cv.Mat) -> pd.DataFrame:
    """Return the filtered image resulting submatrix (actually just a pixel)"""
    # shape[0] is the image height, shape[1] is the image width
    (height, width) = in_img.shape
    # List which will store the submatrices
    sub_list = []

    # For the filtered images, just the central pixel of the original submatrices is needed
    # This means that only in_img[i, j] is needed
    for i in range(0, height):  # loop through rows
        for j in range(0, width):   # loop through columns
            cur_sub = np.array([in_img[i, j]])
            sub_list. append(cur_sub)

    return pd.DataFrame(np.array(sub_list, dtype=np.uint8))
## End of subimage section

## Start of rebuild image section
def rebuildImages(pixels, height: int, width: int) -> list[cv.Mat]:
    """This function rebuilds a list of images"""
    num_pixels = height * width
    img_list = []

    for i in range(0, len(pixels.values), num_pixels):
        cur_pixels = pixels.values[i:(i+num_pixels)]
        cur_img = rebuildSingleImage(cur_pixels, height, width)
        img_list.append(cur_img)

    return img_list

def rebuildSingleImage(pixels: np.array, height: int, width: int) -> cv.Mat:
    """This function rebuilds an image, heightXwidth in size with its pixels"""
    cur_pixel = 0
    rebuilt_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(0, height):  # loop through rows 
        for j in range(0, width):   # loop through columns
            rebuilt_img[i, j] = pixels[cur_pixel]
            cur_pixel += 1

    return rebuilt_img
## End of rebuild image section

def predictionProcessing(pred: np.array) -> pd.DataFrame:
    """This function corrects any out of bound values for np.uint8 and returns the values as a DataFrame"""
    # Checking for underflow and overflow
    for i in range(0, len(pred)):
        if (pred[i] < 0):
            pred[i] = np.array([0])
        elif (pred[i] > 255):
            pred[i] = np.array([255])

    # Converting the predictions into a DataFrame
    return pd.DataFrame(pred).astype(np.uint8)

## Start of image similarity measurement section
def getSSIM(ref: cv.Mat, test: cv.Mat) -> tuple[np.float64, cv.Mat]:
    """This function calculates the structural similarity index, which indicates how similar images ref and test are.
    0 indicates totally dissimilar images
    1 indicates identical images
    It also returns a difference image
    """
    (ssim, diff) = structural_similarity(ref, test, full=True)
    
    # diff is returned as a float array with values in the [0, 1] range
    # Conversion to uint8 is need to be able to show or save the image
    diff = (diff * 255).astype("uint8")

    return (ssim, diff)

def getPSNR(ref: cv.Mat, test: cv.Mat) -> np.float64:
    """Calculates the peak signal to noise ratio of two images
    Lower values (potentially negative) indicate dissimilarity
    Higher values (potentially infinity) indicate similarity
    """
    return peak_signal_noise_ratio(ref, test)

def getNMI(ref: cv.Mat, test: cv.Mat) -> np.float64:
    """Calculates the normalized mutual information of two images
    1 indicates uncorrelated images
    2 indicates correlated images
    """
    return normalized_mutual_information(ref, test)

def getMSE(ref: cv.Mat, test: cv.Mat) -> np.float64:
    """Calculates the mean squared error of two images"""
    return mean_squared_error(ref, test)
## End of image similarity measurement section

# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")