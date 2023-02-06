import os
from time import time
import cv2 as cv
import file_utils as fu
import image_utils as iu
import argparse
from matplotlib import pyplot as plt

def main():
    # TODO: change argument parsing
    parser = argparse.ArgumentParser(description='Apply a filter to a image through a previously trained model')
    parser.add_argument("-m", "--model", type=str, default="trained.pkl", help="Path to trained model")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to image to process")
    args = parser.parse_args()

    file_name = args.image
    model_name = args.model

    if file_name is None:
        raise RuntimeError('No image specified')

    regressor = fu.loadModelPkl(model_name)

    if regressor is None:
        raise RuntimeError('Trained model file not found')

    # Necessary directory and file names
    image_name = os.path.basename(file_name)
    (image_name, image_ext) = os.path.splitext(image_name)
    predict_dir = './prediccion/'
    pred_name = image_name + '-p' + image_ext
    filt_name = image_name + '-f' + image_ext
    diff_name = image_name + '-d' + image_ext

    # Loading original image into memory as a grayscale image
    image = cv.imread(file_name, cv.IMREAD_GRAYSCALE)

    # If image wasn't read, then the file doesn't exist
    if image is None:
        raise FileNotFoundError('Image \'{0}\' not found'.format(file_name))

    # Image characteristics
    (height, width) = image.shape

    cv.imwrite(predict_dir + image_name + image_ext, image)

    # For testing purposes, apply the same transformation the regressor is trying to predict
    iu.gaussianFilter(file_name, predict_dir + filt_name)
    filtered = cv.imread(predict_dir + filt_name, cv.IMREAD_GRAYSCALE)

    # Applying feature engineering
    df_image = iu.getOriginalImgSubmatrices(image)
    
    t0 = time()
    y_pred = regressor.predict(df_image)
    print("Predicting Time:", round(time()-t0, 3), "s")

    # Necessary post-processing
    df_pred = iu.predictionProcessing(y_pred)
    rebuilt_pred = iu.rebuildImages(df_pred, height, width)
    fu.writeImages(predict_dir, pred_name, rebuilt_pred)

    # TODO: show images in imshow for easy comparison
    ssim, diff = iu.getSSIM(filtered, rebuilt_pred[0])
    print('Image SSIM = {0}'.format(round(ssim, 3)))
    cv.imwrite(predict_dir + diff_name, diff)


    f_pred = iu.fourierTransform(rebuilt_pred[0])
    f_actual = iu.fourierTransform(filtered)
    f_mse = iu.getMSE(f_pred, f_actual)

    print('Spectrum MSE = {0}'.format(round(f_mse, 3)))

    plt.subplot(121), plt.imshow(f_pred, cmap = 'gray')
    plt.title('Predicted Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(f_actual, cmap = 'gray')
    plt.title('Actual Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
   main()
