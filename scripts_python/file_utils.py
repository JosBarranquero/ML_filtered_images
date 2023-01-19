import os
import random
import cv2 as cv
import pandas as pd
import numpy as np
import image_utils as iu

def loadImages(original_dir: str, original_files: list[str], filtered_dir: str, filtered_files: list[str], submatrix: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """This function loads the images stored in original_dir + original_files[i] and filtered_dir + filtered_files[i]
    if submatrix is True, the images get divided in submatrices
    """
    df_original = pd.DataFrame()
    df_filtered = pd.DataFrame()

    if not submatrix:
        original_list = []
        filtered_list = []

    for i in range(0, len(original_files)):
        original_path = original_dir + original_files[i]
        filtered_path = filtered_dir + filtered_files[i]

        # Loading the image into memory
        original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
        filtered_img = cv.imread(filtered_path, cv.IMREAD_GRAYSCALE)

        # Getting the physical size of the image (all of them must be the same size)
        (height, width) = original_img.shape

        if submatrix:
            # Applying "feature engineering"
            # Filters are applied convoluting a matrix
            # By getting the array of pixels that get convoluted with the matrix and its result, we may get better predictions
            df_original = pd.concat([df_original, iu.getOriginalImgSubmatrices(original_img)])
            df_filtered = pd.concat([df_filtered, iu.getFilteredImgSubmatrices(filtered_img)])
        else:
            # Converting the image from a heightXwidth matrix to a (height*width) vector
            original_list.append(np.reshape(original_img, newshape=(1, np.product((height, width))))[0])
            filtered_list.append(np.reshape(filtered_img, newshape=(1, np.product((height, width))))[0])

    if not submatrix:
        # Lists to Pandas DataFrame conversion
        df_original = pd.DataFrame(np.array(original_list, dtype=np.uint8))
        df_filtered = pd.DataFrame(np.array(filtered_list, dtype=np.uint8))

    return df_original, df_filtered, height, width

def writeImages(directory: str, filename: str, images: list[cv.Mat]):
    """Save to disk a list of images"""
    for i in range(0, len(images)):
        cv.imwrite((directory + filename).format(i), images[i])

def trainTestSplit(x: list[str], y: list[str], test_size: float, fixed_state: bool = False) -> tuple[list[str], list[str], list[str], list[str]]:
    """This function separates both lists into training and test datasets. x and y are assumed to be the same length
    fix_state param forces a specific order, used for testing purposes
    """
    train_size = 1 - test_size
    x_len = len(x)
    # As it's needed to pick items at random, a random index list is created
    random_index = randomIndexList(x, fixed_state)

    # Using this random index list, the subsets are created
    train_index = random_index[0:int(x_len*train_size)]
    test_index = random_index[int(x_len*train_size):x_len]

    X_train = [x[i] for i in train_index]
    y_train = [y[i] for i in train_index]
    X_test = [x[i] for i in test_index]
    y_test = [y[i] for i in test_index]

    return X_train, X_test, y_train, y_test

def randomIndexList(a_list: list[str], fixed_state: bool) -> list[int]:
    """This funcion returns a list of indices in random order
    fix_state param forces a specific order, used for testing purposes
    """
    max_value = len(a_list)
    
    if fixed_state:
        # Getting a list from max_value to 0, in reverse
        indices = range(max_value-1, -1, -1)
    else:
        # Getting a list of max_value random unrepeated numbers between 0 and max_value
        indices = random.sample(range(0, max_value), max_value)

    return indices
        
def cleanDirectory(directory: str, extension: str = '.bmp'):
    """Deletes a directory contents"""

    # Iterating through the files
    for file in os.listdir(directory):
        # Only counting files ending in extension
        if file.endswith(extension):
            os.remove(directory + file)
    
def fileExists(file_name: str) -> bool:
    """This function checks if a given file exists"""
    return os.path.isfile(file_name)

def fileCount(directory: str, extension: str) -> int:
    """This function counts the files in a directory, given a certain extension"""
    num_files = 0

    # Iterating through the files
    for file in os.listdir(directory):
        # Only counting files ending in extension
        if file.endswith(extension):
            num_files += 1

    return num_files

def fileList(directory: str, extension: str = '.bmp') -> list[str]:
    """This function looks for files in a directory, given a certain extension. Then, it returns them in a list"""
    files = list()

    # Iterating through the files
    for file in sorted(os.listdir(directory)):
        # Only keeping track of files ending in extension
        if os.path.isfile(directory + file) and file.endswith(extension):
            files.append(file)

    return files
    
# Show a message if the script is run by itself
if __name__ == '__main__':
    print("This script is not desingned to be standalone.")