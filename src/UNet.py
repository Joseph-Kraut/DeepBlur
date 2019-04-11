import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

class UNet:
    """
    This class implements a UNet from https://bit.ly/2UAvptW
    """

    def __init__(self, pretrained=False, learning_rate=1e-3):
        """
        Sets up the model for training and inference
        :param pretrained: Whether or not to use pretrained weights
        :param learning_rate: The learning rate for training
        """

        # Printout
        if pretrained:
            print("Building saved UNet model...")
        else:
            print("Building new UNet model...")

        # Build the model
        self.build_model(pretrained, learning_rate)

    def build_model(self, pretrained, learning_rate):
        """
        Builds the model, either from scratch or from pretrained
        :param learning_rate: The learning rate for training
        :param pretrained: Whether or not to pull from pretrained models
        :return: None
        """
        # Cleanup graph
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Placeholders for inputs and labels
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

        # This code is modified from https://bit.ly/2UAvptW
        def upsample_and_concat(x1, x2, output_channels, in_channels):
            pool_size = 2
            deconv_filter = tf.Variable(
                tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
            deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

            deconv_output = tf.concat([deconv, x2], 3)
            deconv_output.set_shape([None, None, None, output_channels * 2])

            return deconv_output

        conv1 = slim.conv2d(self.input_placeholder, 32, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 32, 64)

        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')

        self.output = conv10

        # Load pre-trained weights
        if pretrained:
            # This gets all the weights except all the 10th layer or first layer filters
            pre_trained_variables = [var for var in tf.trainable_variables()
                                     if var.name != "g_conv1_1/weights:0" and not var.name.startswith("g_conv10")]

            # Restore the variables we require
            saver = tf.train.Saver(pre_trained_variables)
            saver.restore(self.sess, "./checkpoints/model.ckpt")

            # Restore the first layer weights by removing one channel (the alpha channel)
            # Pull the value of the first layer weights and remove the last channel
            reader = pywrap_tensorflow.NewCheckpointReader("./checkpoints/model.ckpt")
            first_layer_weights = reader.get_tensor("g_conv1_1/weights")
            first_layer_weights = first_layer_weights[:, :, :3, :]

            # Setup the uninitialized weights
            uninitialized = [var for var in tf.trainable_variables() if (var.name.startswith("g_conv10") or var.name == 'g_conv1_1/weights:0')]
            self.sess.run(tf.initialize_variables(uninitialized))

            # Set the first layer weights equal to the weights we just pulled
            conv1_weights = [var for var in tf.global_variables() if var.name == 'g_conv1_1/weights:0'][0]
            self.sess.run(conv1_weights.assign(first_layer_weights))

            # Build the loss and frozen optimizer
            self.loss = tf.losses.mean_squared_error(self.labels, self.output)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

            # Get the variables that we will train in pretrained model
            trainables = [var for var in tf.trainable_variables() if
                          (var.name.startswith("g_conv9") or var.name.startswith("g_conv10"))]
            trainable_gradients = optimizer.compute_gradients(self.loss, var_list=trainables)
            self.train_op = optimizer.apply_gradients(trainable_gradients)

        else:
            # If we aren't using pre-trained weights then we can just skip this part
            self.sess.run(tf.global_variables_initializer())
            self.loss = tf.losses.mean_squared_error(self.labels, self.output)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        # Init the variables relevant to the optimizer
        optimizer_init = tf.variables_initializer(optimizer.variables())
        self.sess.run(optimizer_init)

    def evaluate(self, inputs, ground_truth):
        """
        Evaluates the model on a set of inputs and outputs
        :param inputs: The input to feed through the network
        :param ground_truth: The output to compare to the network prediction
        :return: A loss value
        """
        feed_dict = {
            self.input_placeholder: inputs,
            self.labels: ground_truth
        }

        return self.sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, inputs):
        """
        Outputs the result of the forward pass of inputs
        :param inputs: The input to forward pass through the network
        :return: The output of the forward pass
        """
        feed_dict = {
            self.input_placeholder: inputs
        }

        return self.sess.run(self.output, feed_dict=feed_dict)

    def train_step(self, inputs, ground_truth):
        """
        Performs an optimizer step given the input and ground truth values
        :param inputs: Inputs to the forward pass
        :param ground_truth: Ground truth labels for the forward pass
        :return: Loss on the training step in question
        """
        feed_dict = {
            self.input_placeholder: inputs,
            self.labels: ground_truth
        }

        loss_value, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
        return loss_value

    def save_model(self):
        """
        Saves the model in the checkpoints folder
        :return: None
        """
        print("Saving model...")
        saver = tf.train.Saver()
        saver.save(self.sess, "./checkpoints/UNet")
