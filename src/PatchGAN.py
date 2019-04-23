import tensorflow as tf
import tensorflow.contrib.slim as slim

class PatchGAN:

    def __init__(self, generator_learning_rate=1e-4, discriminator_learning_rate=1e-5,
                 discriminator_train_steps=1, wasserstein=False, l2_penalty=1e-3):
        """
        Sets up the PatchGAN model
        :param generator_learning_rate: The learning rate to use when optimizing the generator
        :param discriminator_learning_rate: The learning rate to use when optimizing the discriminator
        :param discriminator_train_steps: The number of training steps to make for the discriminator in
        between the training steps for the generator
        :param wasserstein: Whether or not to use teh
        """
        if wasserstein:
            print("Building PatchGAN model with Wasserstein metric...")
        else:
            print("Building PatchGAN Model...")

        self.sess = tf.Session()
        self.build_model(generator_learning_rate, discriminator_learning_rate, wasserstein, l2_penalty)


    def build_model(self, gen_lr, discrim_lr, wasserstein, l2_penalty):
        """
        Builds the computational graph for the PatchGAN implementation
        :param gen_lr: The generator learning rate
        :param discrim_lr: The discriminator learning rate
        :param wasserstein: Whether or not to use the Wasserstein distance metric
        :return: None
        """

        # Build the generator
        with tf.variable_scope("Generator"):
            # The input to the generator
            self.generator_input = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

            # Build the generator graphs
            self.generator_output = self._build_generator(self.generator_input)
            self.generator_labels = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

        # Build the discriminator
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            # The inputs and outputs that are true pairings
            self.discriminator_real_inputs = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)
            self.discriminator_real_labels = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

            # The inputs and outputs that are fake pairings
            # The fake inputs are from a generator output
            self.discriminator_fake_labels = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

            # The discriminator output for the real pairings
            discriminator_real_output = self._build_discriminator(self.discriminator_real_inputs, self.discriminator_real_labels)
            discriminator_fake_output = self._build_discriminator(self.generator_output, self.discriminator_fake_labels)

        with tf.variable_scope("Losses"):
            if wasserstein:
                raise NotImplementedError()
            else:
                # The generator loss
                self.generator_loss = l2_penalty * tf.linalg.norm(self.generator_output - self.generator_labels)
                self.generator_loss += -tf.reduce_sum(tf.log(tf.nn.sigmoid(discriminator_fake_output)))

                # The discriminator loss
                self.discriminator_loss = tf.reduce_sum(tf.nn.sigmoid(discriminator_fake_output))
                self.discriminator_loss -= tf.reduce_sum(tf.log(tf.nn.sigmoid(discriminator_real_output)))


            # Optimizers and train ops
            generator_optimizer = tf.train.AdamOptimizer(gen_lr)
            discriminator_optimizer = tf.train.AdamOptimizer(discrim_lr)

            generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
            discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")

            self.update_generator = generator_optimizer.minimize(self.generator_loss, var_list=generator_vars)
            self.update_discriminator = discriminator_optimizer.minimize(self.discriminator_loss, var_list=discriminator_vars)


        self.sess.run(tf.global_variables_initializer())

    def _build_generator(self, input):
        """
        Builds the graph for the generator
        :param input: The input placeholder to the generator
        :return: The output tensor from the generator forward pass
        """
        # Build a UNet Generator
        # Copied from work in UNet.py
        def upsample_and_concat(x1, x2, output_channels, in_channels):
            pool_size = 2
            deconv_filter = tf.Variable(
                tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
            deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

            deconv_output = tf.concat([deconv, x2], 3)
            deconv_output.set_shape([None, None, None, output_channels * 2])

            return deconv_output

        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.elu, scope='g_conv1_1')
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

        return conv10

    def _build_discriminator(self, input, label):
        """
        Builds the graph for the discriminator
        :param input: The input that was handed to the generator
        :param label: The ground truth image that we are trying to translate to,
        can be an output from the generatord
        :return: The output of the discriminator, a feature map where each pixel is a score
        for the realism of the image translation
        """
        # We reuse the weights to get the same discriminator both times
        # Simple convnet for the discriminator
        lrelu = tf.nn.leaky_relu
        conv = tf.layers.conv2d
        batch_norm = tf.layers.batch_normalization

        output = lrelu(conv(input, 64, 3, name="discriminator_conv1"))
        output = lrelu(batch_norm(conv(output, 128, 3, name="discriminator_conv2")))
        output = lrelu(batch_norm(conv(output, 256, 3, name="discriminator_conv3")))
        output = lrelu(batch_norm(conv(output, 512, 3, name="discriminator_conv4")))

        # This is a feature map that we average over for scoring generator
        return output

    def evaluate(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()


if __name__ == '__main__':
    model = PatchGAN()