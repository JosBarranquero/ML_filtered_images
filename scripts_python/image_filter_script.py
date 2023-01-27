import image_utils as iu
import file_utils as fu

# Processing options (disabled)
# low_filter = False
# high_filter = False
# median_filter = False
# hsobel_filter = False
# vsobel_filter = False
# Processing options (enabled)
low_filter = True
high_filter = True
median_filter = True
hsobel_filter = True
vsobel_filter = True
gaussian_filter = True
canny_filter = True
laplace_filter = True

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