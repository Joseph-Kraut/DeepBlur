""" Data pipeline methods """
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def build_batch(blur_dir, truth_dir, batch_size, resolution=256, validation=False):
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
    print(n)
    print(blur_dir)
    # Allow ourselves to loop infinitely over a dataset
    while True:
        # Shuffle is in-place
        np.random.shuffle(filenames)
        for i in range(0, n, batch_size):
            blur_batch, truth_batch = [], []
            # Python list comp allows going past last iter
            for filename in filenames[i:i+batch_size]:
                with Image.open(os.path.join(blur_dir, filename), 'r') as blurry:
                    imagenp = np.array(blurry)
                    # (,,3) if color
                    if imagenp.shape == (resolution, resolution, 3):
                        blur_batch.append(np.array(blurry))
                with Image.open(os.path.join(truth_dir, filename), 'r') as truthy:
                    imagenp = np.array(truthy)
                    # (,,3) if color
                    if imagenp.shape == (resolution, resolution, 3):
                        truth_batch.append(np.array(truthy))
                    elif validation:
                        truth_batch.append(np.array(truthy))
            yield np.array(blur_batch), np.array(truth_batch)


def train_model(model, train_steps, blur_dir, truth_dir, vblur_dir, vtruth_dir,
                batch_size=16, resolution=256, print_every=50, save=True, graph=False):
    """
    Trains a given model on training data
    :param model: The model to train
    :param train_steps: the number of gradient steps to take in training
    :param print_every: How often to print out the loss
    :param graph: Whether or not to graph a loss vs train step plot at the end
    :return: None
    """
    # Build the batch generator
    batch_generator = build_batch(blur_dir, truth_dir, batch_size, resolution=resolution)
    validation_batch_generator = build_batch(vblur_dir, vtruth_dir, 16, resolution=resolution)
    losses = []
    vlosses = []

    # Loop over the training steps
    with open("training_log.txt", "w+") as logfile:
      try:
        for train_step in range(train_steps):
            if train_step > 0 and train_step % 100 == 0:
                # Run a validation step
                vblurry, vtruth = next(validation_batch_generator)
                vblurry = (vblurry / 127.5) - 1.0
                vtruth = (vtruth / 127.5) - 1.0
                vloss, voutput = model.predict(vblurry, vtruth)
                print(vloss)
                vlosses += [vloss]
                np.save('vresults/inputs', vblurry)
                np.save('vresults/outputs', voutput)
                np.save('vresults/truths', vtruth)

            # Sample a batch
            input_batch, labels_batch = next(batch_generator)
            input_batch = (input_batch / 127.5) - 1.0
            labels_batch = (labels_batch / 127.5) - 1.0
            if input_batch.shape[0] < 10:
              print(input_batch.shape)
            # b&w H4x0rs
            # input_batch = np.reshape(input_batch, (*input_batch.shape, 1))
            # labels_batch = np.reshape(labels_batch, (*labels_batch.shape, 1))

            # Take a train step on this batch
            loss_value = model.train_step(input_batch, labels_batch)

            # Print the loss if desired
            if train_step % print_every == 0:
                print(f"Loss on train step {train_step}: {loss_value}\n")
                logfile.write(f"Loss on train step {train_step}: {loss_value}\n")
                logfile.flush()
            losses += [loss_value]
        except Exception as e:
            logfile.write(str(e))
            logfile.flush()
            raise

        if save:
            model.save_model()

        if graph:
            plt.plot(range(len(losses)), losses)

        logfile.write('Training losses:')
        logfile.write(str(losses))
        logfile.write('Validation losses:')
        logfile.write(str(vlosses))
        logfile.flush()
        return





