from time import time
import file_utils as fu
import image_utils as iu
import regression_utils as ru
from matplotlib import pyplot as plt

# Processing options
regressor = ru.LINEAR
adaboost = False
sub_size = 3

# Pickle options to save and load
save_model_pkl = False
save_images_pkl = False
load_model_pkl = False
load_images_pkl = False

# Directories
original_dir = './originales/'
filtered_dir = './filtradas/'
predict_dir = './prediccion/'

# File extensions
img_ext = '.bmp'
filter_ext = '-low' + img_ext
pred_ext = 'pred-{0}'

if load_images_pkl:
    t0 = time()
    
    height, width, X_train, y_train, X_test, y_test = fu.loadImgPkl()

    if X_test is None:
        raise RuntimeError('Pickle file not found')

    print('Loading Time (pickle):', round(time()-t0, 3), 's')
else:
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
        original_files, filtered_files, test_percent)

    # Measuring load times
    t0 = time()

    # Loading training and test images into memory
    X_train, y_train, height, width = fu.loadImages(original_dir, original_train_files, filtered_dir, filtered_train_files, sub_size=sub_size)
    X_test, y_test, height, width = fu.loadImages(original_dir, original_test_files, filtered_dir, filtered_test_files, sub_size=sub_size)

    print('Loading Time (regular):', round(time()-t0, 3), 's')

    if save_images_pkl:
        fu.saveImgPkl(height, width, X_train, y_train, X_test, y_test)

if load_model_pkl:
    ru.loadRegressor()
else:
    # Creating the regressor
    ru.createRegressor(regressor, adaboost)
    t0 = time()     # To check how long it takes to train
    ru.fitRegressor(X_train, y_train)
    print('Training Time:', round(time()-t0, 3), 's')
    if save_model_pkl:
        ru.saveRegressor()

# Testing the regressor
t0 = time()
y_pred = ru.predict(X_test)
print('Predicting Time:', round(time()-t0, 3), 's')

# Performing necessary processing
df_pred = iu.predictionProcessing(y_pred)

# Rebuild the images into complete images once again
# as df_pred an y_test only contain separated pixels
rebuilt_pred = iu.rebuildImages(df_pred, height, width)
rebuilt_actual = iu.rebuildImages(y_test, height, width)

# List to save difference images
diff_imgs = list()

# Similarity measurement
for i in range(0, len(rebuilt_actual)):
    current_actual = rebuilt_actual[i]
    current_pred = rebuilt_pred[i]
    mse = iu.getMSE(current_actual, current_pred)
    ssim, diff = iu.getSSIM(current_actual, current_pred)
    psnr = iu.getPSNR(current_actual, current_pred)
    nmi = iu.getNMI(current_actual, current_pred)
    diff_imgs.append(diff)
    print('==== Image {0} ===='.format(i))
    print('MSE = {0}'.format(round(mse, 3)))
    print('SSIM = {0}'.format(round(ssim, 3)))
    print('PSNR = {0} dB'.format(round(psnr, 3)))
    print('NMI = {0}'.format(round(nmi, 3)))

# Cleaning the output directory
fu.cleanDirectory(predict_dir)

# Save the predictions and actual images to disk for manual comparison
fu.writeImages(predict_dir, pred_ext + img_ext, rebuilt_pred)
fu.writeImages(predict_dir, pred_ext + '-actual' + img_ext, rebuilt_actual)
fu.writeImages(predict_dir, pred_ext + '-diff' + img_ext, diff_imgs)

# Fourier transform of the first image
first_pred = rebuilt_pred[0]
first_actual = rebuilt_actual[0]
f_pred = iu.fourierTransform(first_pred)
f_actual = iu.fourierTransform(first_actual)
f_mse = iu.getMSE(f_pred, f_actual)

print('===================')
print('Spectrum MSE = {0}'.format(round(f_mse, 3)))

# Show the first image (prediction and actual) alongside their frequency spectrums
plt.subplot(221), plt.imshow(first_pred, cmap = 'gray')
plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(first_actual, cmap = 'gray')
plt.title('Actual Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(f_pred, cmap = 'gray')
plt.title('Predicted Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(f_actual, cmap = 'gray')
plt.title('Actual Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()