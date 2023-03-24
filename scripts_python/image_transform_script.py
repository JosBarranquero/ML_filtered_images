import image_utils as iu
import file_utils as fu
import random

# Processing options
rotation = True
scaling = True
translation = True
total = True

input_path = './originales/'
output_path  = './originales/ampliado/'

# Counting the images to process
num_images = fu.fileCount(input_path, '.bmp')

# Processing images one by one
for i in range(1, num_images + 1):
    input = '{0}imagen{1}.bmp'.format(input_path, i)

    if rotation:
        output_rotation = '{0}imagen{1}-rotated.bmp'.format(output_path, i)
        angle = 0
        while angle == 0:
            angle = random.uniform(-3, 3)
        if (not fu.fileExists(output_rotation)):     # Only process if the output file doesn't already exist
            iu.rotateImage(input, output_rotation, angle)

    if scaling:
        output_scaling = '{0}imagen{1}-scaled.bmp'.format(output_path, i)
        scale_factor = random.uniform(0.95, 1.15)
        if (not fu.fileExists(output_scaling)):     # Only process if the output file doesn't already exist
            iu.scaleImage(input, output_scaling, scale_factor)

    if translation:
        output_translation = '{0}imagen{1}-translated.bmp'.format(output_path, i)
        tx = 0
        ty = 0
        while tx == 0 and ty == 0:
            tx = random.randint(-4, 4)
            ty = random.randint(-4, 4)
        if (not fu.fileExists(output_translation)):     # Only process if the output file doesn't already exist
            iu.translateImage(input, output_translation, tx, ty)

    if total:
        output_total = '{0}imagen{1}-total.bmp'.format(output_path, i)
        if (not fu.fileExists(output_total)):     # Only process if the output file doesn't already exist
            iu.totalTransformation(input, output_total, angle, scale_factor, tx, ty)