""" Data pipeline methods """
import numpy as np
import os
from PIL import Image


def build_batch(blur_dir, truth_dir, batch_size):
    """
    Generates a mini-batch of numpy arrays; uses a generator to truly create epochs.
    IMPORTANT: Assumes truth_dir contains exactly the same set of file names as truth_dir, where the
    files are the corresponding truth image samples. 

    Returns:
        data_batch = List[np.array], of size batch_size, containing blurred image pixel arrays.
        truth_batch = List[np.array], of size batch_size, containing the corresponding ground truths.
    """
    filenames = os.listdir(blur_dir)
    n = len(filenames)
    # Allow ourselves to loop infinitely over a dataset
    while True:
        # Shuffle is in-place
        np.random.shuffle(filenames)
        for i in range(0, n, batch_size):
            blur_batch, truth_batch = [], []
            # Python list comp allows going past last iter
            for filename in filenames[i:i+batch_size]:
                with Image.open(os.path.join(blur_dir, filename), 'r') as blurry:
                    blur_batch.append(np.array(blurry))
                with Image.open(os.path.join(truth_dir, filename), 'r') as truthy:
                    truth_batch.append(np.array(truthy))
            yield blur_batch, truth_batch
        




