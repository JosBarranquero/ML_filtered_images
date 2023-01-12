from time import time
from math import sqrt
import utilities as u
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Pickle filenames and options to save and load
original_pkl = 'original-eng.pkl'
filtered_pkl = 'filtered-eng.pkl'
save_to_pkl = True
load_fr_pkl = True

# Directories
original_dir = '../originales/'
filtered_dir = '../filtradas/'
predict_dir = '../prediccion/'

# File extensions
img_ext = '.bmp'
filter_ext = '-low' + img_ext
pred_ext = 'pred-{0}'

# Image characteristics
width = 0
height = 0

if (load_fr_pkl and (u.file_exists(original_pkl) and u.file_exists(filtered_pkl))):
    # Measuring load times
    t0 = time()
    df_original = pd.read_pickle(original_pkl)
    df_filtered = pd.read_pickle(filtered_pkl)

    print("Loading Time (pickle):", round(time()-t0, 3), "s")

    # Assuming square images
    width = int(sqrt(len(df_original.columns)))
    height = width
else:
    # Lists in which the original and filtered images will be stored for processing
    original_list = []
    filtered_list = []

    # Recovering the file names to process
    original_files = u.file_list(original_dir, img_ext)
    filtered_files = u.file_list(filtered_dir, filter_ext)

    # Check the number of files
    if (len(original_files) != len(filtered_files)):
        raise RuntimeError('Number of originals and filtered images not matching')

    # Measuring load times
    t0 = time()

    # First image is processed manually
    i = 0
    original_path = original_dir + original_files[i]
    filtered_path = filtered_dir + filtered_files[i]

    # Loading the image into memory
    original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    filtered_img = cv.imread(filtered_path, cv.IMREAD_GRAYSCALE)

    # Manually applying "feature engineering"
    # Filters are applied convoluting a matrix
    # By getting the array of pixels that get convoluted with the matrix and its result, we may get better predictions
    original_img = np.array([original_img[0,0], original_img[0,1], original_img[0,2], original_img[1,0], original_img[1,1], original_img[1,2], original_img[2,0], original_img[2,1], original_img[2,2]])
    filtered_img = np.array(filtered_img[1,1])

    # Getting the physical size of the image (all of them must be the same size)
    (height, width) = (3,3)

    # Converting the image from a heightXwidth matrix to a (height*width) vector
    original_list.append(original_img)
    filtered_list.append(filtered_img)

    # The rest of images get processed automatically
    for i in range(1, len(original_files)):
        original_path = original_dir + original_files[i]
        filtered_path = filtered_dir + filtered_files[i]

        # Loading the image into memory
        original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
        filtered_img = cv.imread(filtered_path, cv.IMREAD_GRAYSCALE)

        original_img = np.array([original_img[0,0], original_img[0,1], original_img[0,2], original_img[1,0], original_img[1,1], original_img[1,2], original_img[2,0], original_img[2,1], original_img[2,2]])
        filtered_img = np.array(filtered_img[1,1])

        # Converting the image from a heightXwidth matrix to a (height*width) vector
        original_list.append(original_img)
        filtered_list.append(filtered_img)

    # Lists to Pandas DataFrame conversion
    df_original = pd.DataFrame(np.array(original_list, dtype=np.uint8))
    df_filtered = pd.DataFrame(np.array(filtered_list, dtype=np.uint8))

    print("Loading Time (regular):", round(time()-t0, 3), "s")

    # Saving to a pickle file
    if (save_to_pkl):
        df_original.to_pickle(original_pkl)
        df_filtered.to_pickle(filtered_pkl)

# Creation of the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df_original, df_filtered, test_size=0.10)
# Manual creation of the datasets
# X_train = df_original.head(75)
# X_test = df_original.tail(25)
# y_train = df_filtrada.head(75)
# y_test = df_filtrada.tail(25)

# Linear Regressor Training
regressor = LinearRegression()
# Regression Tree Training
# regressor = svm.SVR(kernel='linear', C=100.0)
t0 = time()     # To check how long it takes to train
regressor.fit(X_train, y_train)
print("Training Time:", round(time()-t0, 3), "s")

# Testing the regressor
t0 = time()
y_pred = regressor.predict(X_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Mean Squared Error calculation
# TODO: find new error measurements
mse = mean_squared_error(y_pred, y_test)

# Converting the predictions into real viewable images
for i in range (0, len(y_pred)):
    # Converting from a (height*width) vector back to a heightXwidth matrix
    # pred_img = np.reshape(y_pred[i], newshape=(height, width))
    pred_img = y_pred[i]
    pred_img = pred_img.astype(np.uint8)
    
    # imagen_real = y_test.loc[i+75,:]
    # imagen_real = np.reshape(imagen_real.to_numpy(dtype='uint8'), newshape=(height, width))

    cv.imwrite((predict_dir + pred_ext + img_ext).format(i), pred_img)
    # cv.imwrite('../prediccion/prediccion-{0}-real.bmp'.format(i), imagen_real)

# Getting the actual images
indices = y_test.index
for index, value in enumerate(indices):
    actual_img = y_test.loc[value,:]
    actual_img = np.array(actual_img, dtype=np.uint8)
    # imagen_real = np.reshape(imagen_real.to_numpy(dtype='uint8'), newshape=(height, width))
    
    cv.imwrite((predict_dir + pred_ext + '-actual' + img_ext).format(index), actual_img)

print("MSE = {0}".format(mse))