""" Data pipeline methods """
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


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
            yield np.array(blur_batch), np.array(truth_batch)


def train_model(model, train_steps, blur_dir, truth_dir, batch_size=16, print_every=1, save=True, graph=False):
    """
    Trains a given model on training data
    :param model: The model to train
    :param train_steps: the number of gradient steps to take in training
    :param print_every: How often to print out the loss
    :param graph: Whether or not to graph a loss vs train step plot at the end
    :return: None
    """
    # Build the batch generator
    batch_generator = build_batch(blur_dir, truth_dir, batch_size)
    losses = []

    # Loop over the training steps
    for train_step in range(train_steps):
        # Sample a batch
        input_batch, labels_batch = next(batch_generator)
        # H4x0rs
        input_batch = np.reshape(input_batch, (*input_batch.shape, 1))
        labels_batch = np.reshape(input_batch, (*labels_batch.shape, 1))
        print(input_batch.shape)
        # Take a train step on this batch
        loss_value = model.train_step(input_batch, labels_batch)

        # Possibly print the loss
        if train_step % print_every == 0:
            print(f"Loss on train step {train_step}: {loss_value}")

        losses += [loss_value]

    if save:
        model.save_model()

    if graph:
        plt.plot(range(len(losses)), losses)




        




