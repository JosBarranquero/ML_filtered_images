from time import time
from math import sqrt
import file_utils as fu
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Pickle filenames and options to save and load
original_pkl = 'original.pkl'
filtered_pkl = 'filtered.pkl'
save_to_pkl = True
load_fr_pkl = True

# Directories
original_dir = './originales/'
filtered_dir = './filtradas/'
predict_dir = './prediccion/'

# File extensions
img_ext = '.bmp'
filter_ext = '-low' + img_ext
pred_ext = 'pred-{0}'

# Image characteristics
width = 0
height = 0

if (load_fr_pkl and (fu.fileExists(original_pkl) and fu.fileExists(filtered_pkl))):
    # Measuring load times
    t0 = time()
    df_original = pd.read_pickle(original_pkl)
    df_filtered = pd.read_pickle(filtered_pkl)

    print("Loading Time:", round(time()-t0, 3), "s")

    # Assuming square images
    width = int(sqrt(len(df_original.columns)))
    height = width
else:
    # Lists in which the original and filtered images will be stored for processing
    original_list = []
    filtered_list = []

    # Recovering the file names to process
    original_files = fu.fileList(original_dir, img_ext)
    filtered_files = fu.fileList(filtered_dir, filter_ext)

    # Check the number of files
    if (len(original_files) != len(filtered_files)):
        raise RuntimeError('Number of originals and filtered images not matching')

    # Measuring load times
    t0 = time()

    # Loading original and filtered images into memory
    df_original, df_filtered, height, width = fu.loadImages(original_dir, original_files, filtered_dir, filtered_files, submatrix=False)
    print("Loading Time:", round(time()-t0, 3), "s")

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
# Support Vector Regression
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
    pred_img = np.reshape(y_pred[i], newshape=(height, width))
    pred_img = pred_img.astype(np.uint8)
    
    # imagen_real = y_test.loc[i+75,:]
    # imagen_real = np.reshape(imagen_real.to_numpy(dtype='uint8'), newshape=(height, width))

    cv.imwrite((predict_dir + pred_ext + img_ext).format(i), pred_img)
    # cv.imwrite('../prediccion/prediccion-{0}-real.bmp'.format(i), imagen_real)

# Getting the actual images
indices = y_test.index
for index, value in enumerate(indices):
    actual_img = y_test.loc[value,:]
    actual_img = np.reshape(actual_img.to_numpy(dtype='uint8'), newshape=(height, width))
    
    cv.imwrite((predict_dir + pred_ext + '-actual' + img_ext).format(index), actual_img)

print("MSE = {0}".format(mse))