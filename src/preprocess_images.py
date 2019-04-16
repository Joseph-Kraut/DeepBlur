import numpy as np
import os
from PIL import Image
import scipy.signal as signal
from math import pi, e
from os import listdir
from os.path import join, isfile
import re

def gaussian_blur_kernel(kernel_size, standard_deviation):
    """
    Generates a gaussian blur filter of type numpy array
    :param kernel_size: The desired dimension of the filter (odd number)
    :return: The kernel as a numpy array
    """
    # The location of the kernel center
    kernel_center = [int(kernel_size / 2), int(kernel_size / 2)]

    kernel = np.zeros((kernel_size, kernel_size))

    # Loop over the elements in the array
    for row in range(kernel.shape[0]):
        for column in range(kernel.shape[1]):
            # Compute the gaussian value at this point
            exponent = -((row - kernel_center[0]) ** 2 + (column - kernel_center[0]) ** 2) / (2 * standard_deviation ** 2)
            z_score = e ** exponent / (2 * pi * standard_deviation ** 2)

            # Add this z_score to the kernel
            kernel[row][column] = z_score

    # Normalize the kernel to become a convex combination before returning
    kernel = kernel / np.sum(kernel)

    return kernel


def blur(image, number_filters=10, standard_deviation=50):
    """
    Blurs the images by passing multiple gaussian blur filters over and averaging
    :param image: A numpy array corresponding to the image
    :param number_filters: The number of randomly selected blur filters to convolve over the image
    :return: A tuple containing
        - The blurred image
        - The kernels used to blur stacked depth wise
    """
    filter_sizes = [3, 5, 7, 9, 11]
    filter_probabilities = [0.45, 0.45, 0.1]

    # Loop over the number of filters
    for _ in range(number_filters):
        # Select a filter size
        filter_size = np.random.choice(filter_sizes)
        print(f"Blurring with kernel of size: {filter_size}")

        # Generate a gaussian blur filter of size filter_size x filter_size
        kernel = gaussian_blur_kernel(filter_size, standard_deviation)

        # Stack the kernel along the channel dimension
        # Blur each channel
        channels = []
        for channel_index in range(image.shape[2]):
            # Get the current channel
            channel = image[:, :, channel_index]
            # Blur it
            channel = signal.convolve(channel, kernel, mode='same')
            # Add the channel to the list of blurred channels
            channels += [channel]


        # Reform blurred channels into image
    image = np.transpose(np.array(channels), (1, 2, 0))

    return image.astype(int)


def get_samples(filename, sample_res, max_stride, should_blur=False, output_dir='samples'):
    """
    Gets the samples of size sample_res x sample_res from the image specified
    :param filename: The local path to the image (i.e. './path/to/image.png')
    :param sample_res: The size we wish to sample. Our samples will then be of
    dimension (sample_res, sample_res)
    :param max_stride: The maximum amount by which we stride in between samples both horizontally
    and vertically. The only time we stride less than max_stride is if max_stride would go over the border
    :param should_blur: A boolean as to whether we should blur the image. If not, assume the directory has
    a blurred image with the same filename that we can pull from
    :return: None, save the files to a desired output directory
    """
    with Image.open(filename, 'r') as image:
        pixels = np.array(image)
        v_res = pixels.shape[0]
        h_res = pixels.shape[1]

        if should_blur:
            pixels = blur(pixels)
        hbound, vbound = h_res - sample_res, v_res - sample_res
        horizontal_starts = [i for i in range(hbound) if i % max_stride == 0]
        vertical_starts = [i for i in range(vbound) if i % max_stride == 0]
        # Catch the trailing final samples if necessary
        if hbound not in horizontal_starts:
            horizontal_starts.append(hbound)
        if vbound not in vertical_starts:
            vertical_starts.append(vbound)

        sample_num = 0
        for h in horizontal_starts:
            for v in vertical_starts:
                # Following StackOverflow on dtype here
                sample = np.array(pixels[v:v+sample_res, h:h+sample_res], dtype=np.uint8)
                output_filename = re.split("/|_", os.path.splitext(filename)[0])[-1]
                with Image.fromarray(sample) as sample_image:
                    sample_image.save(join(output_dir,
                                    '{0}_{1}res_{2}.png'.format(output_filename, sample_res, sample_num)),
                                    'PNG')
                    sample_num += 1

        print('Sampling for image {0} complete'.format(filename))
    return


def main():
    """
    Iterates over images in dataset to process them
    :return: None
    """
    ALREADY_BLURRED = True #Set to true if you want to blur new images
    SAMPLE_RES = 300 #The resolution (side length) of the square patches
    MAX_STRIDE = 180 #How much each sample overlaps by determines overlap
    CLEAR_SAVE_DIR = True # Deletes pre-existing files in save directory

    sharp_img_dir = "../data/raw_images/ground_truth" #Assumes ground truth files or unblurred imgs stored here
    blurred_img_dir = "../data/raw_images/blurry" #Assumes blurred files (if already blurred) stored here
    #NOTE: blurred image filenames should end (last chars after a "/" or "_") with the same name as unblurred images

    blur_save_dir = "../data/labelled_patches/blurred" #Where blurred images should be saved
    truth_save_dir = "../data/labelled_patches/sharp" #Where unblurred images should be saved

    if CLEAR_SAVE_DIR:
        for filename in listdir(blur_save_dir):
            os.unlink(os.path.join(blur_save_dir, filename))
        for filename in listdir(truth_save_dir):
            os.unlink(os.path.join(truth_save_dir, filename))

    if ALREADY_BLURRED:
        blurred_img_filenames = [join(blurred_img_dir, f) for f in sorted(listdir(blurred_img_dir)) if isfile(join(blurred_img_dir, f))]
        sharp_img_filenames = [join(sharp_img_dir, f) for f in sorted(listdir(sharp_img_dir)) if isfile(join(sharp_img_dir, f))]
        for sharp_img, blurred_img in zip(sharp_img_filenames, blurred_img_filenames):
            get_samples(sharp_img, SAMPLE_RES, MAX_STRIDE, should_blur=False, output_dir=truth_save_dir)
            get_samples(blurred_img, SAMPLE_RES, MAX_STRIDE, should_blur=False, output_dir=blur_save_dir)
    else:
        sharp_img_filenames = [join(sharp_img_dir, f) for f in listdir(sharp_img_dir) if isfile(join(sharp_img_dir, f))]
        for sharp_img in sharp_img_filenames:
            get_samples(sharp_img, SAMPLE_RES, MAX_STRIDE, should_blur=False, output_dir=truth_save_dir)
            get_samples(sharp_img, SAMPLE_RES, MAX_STRIDE, should_blur=True, output_dir=blur_save_dir)



if __name__ == "__main__":
    main()
