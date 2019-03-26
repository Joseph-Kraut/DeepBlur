import tensorflow as tf
import tensorflow.contrib.slim as slim

class UNet():
    """
    This class implements a UNet from https://bit.ly/2UAvptW
    """

    def __init__(self, pretrained=False, learning_rate=1e-3):
        """
        Sets up the model for training and inference
        :param pretrained:
        :param learning_rate:
        """
        # Make a call to the build_model
        self.sess = tf.Session()

        # Printout
        if pretrained:
            print("Building saved UNet model...")
        else:
            print("Building new UNet model...")

        self.build_model(pretrained, learning_rate)


    def build_model(self, pretrained, learning_rate):
        """
        Builds the model, either from scratch or from pretrained
        :param learning_rate: The learning rate for training
        :param pretrained: Whether or not to pull from pretrained models
        :return: None
        """
        # Placeholders for inputs and labels
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])

        # This code is modified from
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

        self.loss = tf.losses.mean_squared_error(self.labels, self.output)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        if pretrained:
            raise NotImplementedError()
        else:
            self.sess.run(tf.global_variables_initializer())

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

