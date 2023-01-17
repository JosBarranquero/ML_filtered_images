from time import time
import file_utils as fu
import image_utils as iu
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Pickle filenames and options to save and load
original_pkl = 'original-eng.pkl'
filtered_pkl = 'filtered-eng.pkl'
save_to_pkl = False
load_fr_pkl = False

# Directories
original_dir = './originales/'
filtered_dir = './filtradas/'
predict_dir = './prediccion/'

# File extensions
img_ext = '.bmp'
filter_ext = '-high' + img_ext
pred_ext = 'pred-{0}'

# Image characteristics
width = 0
height = 0

# Recovering the file names to process
original_files = fu.fileList(original_dir, img_ext)
filtered_files = fu.fileList(filtered_dir, filter_ext)

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

# Getting the physical size of the image (all of them must be the same size)
(height, width) = original_img.shape

# Applying "feature engineering"
# Filters are applied convoluting a matrix
# By getting the array of pixels that get convoluted with the matrix and its result, we may get better predictions
df_original = iu.getOriginalImgSubmatrices(original_img)
df_filtered = iu.getFilteredImgSubmatrices(filtered_img)

for i in range(1, len(original_files)):
    original_path = original_dir + original_files[i]
    filtered_path = filtered_dir + filtered_files[i]

    # Loading the image into memory
    original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    filtered_img = cv.imread(filtered_path, cv.IMREAD_GRAYSCALE)

    df_original = pd.concat([df_original, iu.getOriginalImgSubmatrices(original_img)])
    df_filtered = pd.concat([df_filtered, iu.getFilteredImgSubmatrices(filtered_img)])

print("Loading Time (regular):", round(time()-t0, 3), "s")

# Creation of the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df_original, df_filtered, test_size=0.15)

# Linear Regressor Training
regressor = LinearRegression()
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
    
    # Checking for underflow and overflow
    if (pred_img < 0):
        pred_img = np.array([0])
    elif (pred_img > 255):
        pred_img = np.array([255])

    pred_img = pred_img.astype(np.uint8)

    cv.imwrite((predict_dir + pred_ext + img_ext).format(i), pred_img)

# Getting the actual images
indices = y_test.index
for index, value in enumerate(indices):
    actual_img = y_test.loc[value,:]    # TODO: find another way to find in y_test
    actual_img = np.array(actual_img, dtype=np.uint8)
    
    cv.imwrite((predict_dir + pred_ext + '-actual' + img_ext).format(index), actual_img)

print("MSE = {0}".format(mse))