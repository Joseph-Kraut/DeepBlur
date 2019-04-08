import numpy as np
from PIL import Image
import scipy.signal as signal
from math import pi, e

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


def get_samples(filename, sample_res, max_stride, should_blur=False):
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
        pixels = list(image.getdata())
        v_res, h_res = image.size
        pixel_dimension = len(pixels[0])
        pixels = np.reshape(pixels, (v_res, h_res, pixel_dimension))
        if should_blur:
            pixels, _ = blur(pixels)
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
                with Image.fromarray(sample) as sample_image:
                    # I dislike hardcoding an output directory and think it should be
                    # a function param
                    sample_image.save('samples/{0}_{1}res_{2}.png'.format(filename, sample_res, sample_num))
                    sample_num += 1

        print('Sampling for image {0} complete'.format(filename))
    return


def main():
    """
    Iterates over images in dataset to process them
    :return: None
    """
    
    raise NotImplementedError()


if __name__ == "__main__":
    main()
