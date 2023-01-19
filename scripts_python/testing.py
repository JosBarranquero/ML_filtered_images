import getopt
import os
import sys
from time import time
import cv2 as cv
import file_utils as fu
import image_utils as iu

def main(argv):
    script_name = sys.argv[0]
    help = script_name + ' [-m <trainedModel>] -i <image>'
    file_name = None
    model_name = None
    try:
        opts, args = getopt.getopt(argv,"hi:m:",["imgfile=","modfile="])
    except getopt.GetoptError:
        print(help)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (help)
            sys.exit()
        elif opt in ("-i", "--image"):
            file_name = arg
        elif opt in ("-m", "--model"):
            model_name = arg

    if file_name is None:
        raise RuntimeError('No image specified')

    if model_name is None:  # default file
        regressor = fu.loadModelPkl()
    else:
        regressor = fu.loadModelPkl(model_name)

    if regressor is None:
        raise RuntimeError('Trained model file not found')

    # Necessary directory and file names
    image_name = os.path.basename(file_name)
    (image_name, image_ext) = os.path.splitext(image_name)
    predict_dir = './prediccion/'
    pred_name = image_name + '-p' + image_ext
    filt_name = image_name + '-f' + image_ext

    # Loading original image into memory as a grayscale image
    image = cv.imread(file_name, cv.IMREAD_GRAYSCALE)

    # Image characteristics
    (height, width) = image.shape

    cv.imwrite(predict_dir+image_name+image_ext, image)

    # For testing purposes, apply the same transformation the regressor is trying to predict
    iu.hSobelFilter(file_name, predict_dir + filt_name)

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

if __name__ == "__main__":
   main(sys.argv[1:])