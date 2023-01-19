from time import time
import file_utils as fu
import image_utils as iu
from sklearn.linear_model import LinearRegression
from sklearn import svm
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

# Recovering the file names to process
original_files = fu.fileList(original_dir, img_ext)
filtered_files = fu.fileList(filtered_dir, filter_ext)

# Check the number of files
if (len(original_files) != len(filtered_files)):
    raise RuntimeError('Number of originals and filtered images not matching')

# Separate images in training and test sets
test_percent = 0.05
original_train_files, original_test_files, filtered_train_files, filtered_test_files = fu.trainTestSplit(
    original_files, filtered_files, test_percent, fixed_state=False)

# Measuring load times
t0 = time()

# Loading training and test images into memory
X_train, y_train, height, width = fu.loadImages(original_dir, original_train_files, filtered_dir, filtered_train_files)
X_test, y_test, height, width = fu.loadImages(original_dir, original_test_files, filtered_dir, filtered_test_files)

print("Loading Time (regular):", round(time()-t0, 3), "s")

# Linear Regressor Training
regressor = LinearRegression()
# Support Vector Regression
# regressor = svm.LinearSVR(C=5.0)
t0 = time()     # To check how long it takes to train
regressor.fit(X_train, y_train)
print("Training Time:", round(time()-t0, 3), "s")

# Testing the regressor
t0 = time()
y_pred = regressor.predict(X_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Performing necessary processing
df_pred = iu.predictionProcessing(y_pred)

# Mean Squared Error calculation
# TODO: find new error measurements
mse = mean_squared_error(df_pred, y_test)
print("MSE = {0}".format(mse))

# Rebuild the images into complete images once again
# as df_pred an y_test only contain separated pixels
rebuilt_pred = iu.rebuildImages(df_pred, height, width)
rebuilt_actual = iu.rebuildImages(y_test, height, width)

# Cleaning the output directory
fu.cleanDirectory(predict_dir)

# Save the predictions and actual images to disk for manual comparison
fu.writeImages(predict_dir, pred_ext + img_ext, rebuilt_pred)
fu.writeImages(predict_dir, pred_ext + '-actual' + img_ext, rebuilt_actual)