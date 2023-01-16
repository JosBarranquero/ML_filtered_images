from time import time
from math import sqrt
import file_utils as fu
import image_utils as iu
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
save_to_pkl = False
load_fr_pkl = False

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

# First image is processed manually
i = 0
original_path = original_dir + original_files[i]
filtered_path = filtered_dir + filtered_files[i]

# Loading the image into memory
original_img = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
filtered_img = cv.imread(filtered_path, cv.IMREAD_GRAYSCALE)

iu.getOriginalImgSubmatrices(original_img)