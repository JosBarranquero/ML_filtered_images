import filters as f
import utilities as u

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

input_path = './originales/'
output_path  = './filtradas/'

# Counting the images to process
num_images = u.file_count(input_path, '.bmp')

# Processing images one by one
for i in range(1, num_images + 1):
    input = '{0}imagen{1}.bmp'.format(input_path, i)

    if low_filter:
        output_low = '{0}imagen{1}-low.bmp'.format(output_path, i)
        if (not u.file_exists(output_low)):     # Only process if the output file doesn't already exist
            f.low_pass_filter(input, output_low)

    if high_filter:
        output_high = '{0}imagen{1}-high.bmp'.format(output_path, i)
        if (not u.file_exists(output_high)):     # Only process if the output file doesn't already exist
            f.high_pass_filter(input, output_high)

    if median_filter:
        output_median = '{0}imagen{1}-median.bmp'.format(output_path, i)
        if (not u.file_exists(output_median)):     # Only process if the output file doesn't already exist
            f.median_filter(input, output_median)

    if hsobel_filter:
        output_hsobel = '{0}imagen{1}-hsobel.bmp'.format(output_path, i)
        if (not u.file_exists(output_hsobel)):     # Only process if the output file doesn't already exist
            f.hsobel_filter(input, output_hsobel)

    if vsobel_filter:
        output_vsobel = '{0}imagen{1}-vsobel.bmp'.format(output_path, i)
        if (not u.file_exists(output_vsobel)):     # Only process if the output file doesn't already exist
            f.vsobel_filter(input, output_vsobel)