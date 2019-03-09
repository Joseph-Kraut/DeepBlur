def blur(image):
    """
    Blurs the images by passing multiple gaussian blur filters over and averaging
    :param image: A numpy array corresponding to the image
    :return: A tuple containing
        - The blurred image
        - The kernels used to blur stacked depth wise
    """
    raise NotImplementedError()

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
    raise NotImplementedError()

def main():
    """
    Iterates over images in dataset to process them
    :return: None
    """
    raise NotImplementedError()


if __name__ == "__main__":
    main()