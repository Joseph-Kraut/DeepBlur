from preprocess_images import get_samples

filename = 'nature.jpg'

get_samples(filename, 50, 50, should_blur=True, output_dir='blurry')
get_samples(filename, 50, 50, should_blur=False, output_dir='truthy')
