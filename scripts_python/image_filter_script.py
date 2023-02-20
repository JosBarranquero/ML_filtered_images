import image_utils as iu
import file_utils as fu

# Processing options
low_filter = True
high_filter = True
median_filter = True
hsobel_filter = True
vsobel_filter = True
gaussian_filter = True
canny_filter = True
laplace_filter = True
bilateral_filter = True
motionblur_filter = True
sharpen_filter = True
emboss_filter = True
custom_filter = True
equalize_hist = True

input_path = './originales/'
output_path  = './filtradas/'

# Counting the images to process
num_images = fu.fileCount(input_path, '.bmp')

# Processing images one by one
for i in range(1, num_images + 1):
    input = '{0}imagen{1}.bmp'.format(input_path, i)

    if low_filter:
        output_low = '{0}imagen{1}-low.bmp'.format(output_path, i)
        if (not fu.fileExists(output_low)):     # Only process if the output file doesn't already exist
            iu.lowPassFilter(input, output_low)

    if high_filter:
        output_high = '{0}imagen{1}-high.bmp'.format(output_path, i)
        if (not fu.fileExists(output_high)):     # Only process if the output file doesn't already exist
            iu.highPassFilter(input, output_high)

    if median_filter:
        output_median = '{0}imagen{1}-median.bmp'.format(output_path, i)
        if (not fu.fileExists(output_median)):     # Only process if the output file doesn't already exist
            iu.medianFilter(input, output_median)

    if hsobel_filter:
        output_hsobel = '{0}imagen{1}-hsobel.bmp'.format(output_path, i)
        if (not fu.fileExists(output_hsobel)):     # Only process if the output file doesn't already exist
            iu.hSobelFilter(input, output_hsobel)

    if vsobel_filter:
        output_vsobel = '{0}imagen{1}-vsobel.bmp'.format(output_path, i)
        if (not fu.fileExists(output_vsobel)):     # Only process if the output file doesn't already exist
            iu.vSobelFilter(input, output_vsobel)

    if gaussian_filter:
        output_gaussian = '{0}imagen{1}-gaussian.bmp'.format(output_path, i)
        if (not fu.fileExists(output_gaussian)):     # Only process if the output file doesn't already exist
            iu.gaussianFilter(input, output_gaussian)

    if canny_filter:
        output_canny = '{0}imagen{1}-canny.bmp'.format(output_path, i)
        if (not fu.fileExists(output_canny)):     # Only process if the output file doesn't already exist
            iu.cannyFilter(input, output_canny)

    if laplace_filter:
        output_laplace = '{0}imagen{1}-laplace.bmp'.format(output_path, i)
        if (not fu.fileExists(output_laplace)):     # Only process if the output file doesn't already exist
            iu.laplaceFilter(input, output_laplace)

    if bilateral_filter:
        output_bilateral = '{0}imagen{1}-bilateral.bmp'.format(output_path, i)
        if (not fu.fileExists(output_bilateral)):     # Only process if the output file doesn't already exist
            iu.bilateralFilter(input, output_bilateral)

    if motionblur_filter:
        output_motion = '{0}imagen{1}-motion.bmp'.format(output_path, i)
        if (not fu.fileExists(output_motion)):     # Only process if the output file doesn't already exist
            iu.motionBlurFilter(input, output_motion)

    if sharpen_filter:
        output_sharpen = '{0}imagen{1}-sharpen.bmp'.format(output_path, i)
        if (not fu.fileExists(output_sharpen)):     # Only process if the output file doesn't already exist
            iu.sharpenFilter(input, output_sharpen)

    if emboss_filter:
        output_emboss = '{0}imagen{1}-emboss.bmp'.format(output_path, i)
        if (not fu.fileExists(output_emboss)):     # Only process if the output file doesn't already exist
            iu.embossFilter(input, output_emboss)

    if custom_filter:
        output_custom = '{0}imagen{1}-custom.bmp'.format(output_path, i)
        if (not fu.fileExists(output_custom)):     # Only process if the output file doesn't already exist
            iu.customFilter(input, output_custom)

    if equalize_hist:
        output_equalize = '{0}imagen{1}-equalize.bmp'.format(output_path, i)
        if (not fu.fileExists(output_equalize)):     # Only process if the output file doesn't already exist
            iu.histogramEq(input, output_equalize)