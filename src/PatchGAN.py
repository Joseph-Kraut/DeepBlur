import tensorflow as tf

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

            # Build the generator graph
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
        raise NotImplementedError()

    def _build_discriminator(self, input, label):
        """
        Builds the graph for the discriminator
        :param input: The input that was handed to the generator
        :param label: The ground truth image that we are trying to translate to,
        can be an output from the generator
        :return: The output of the discriminator, a feature map where each pixel is a score
        for the realism of the image translation
        """
        # We reuse the weights to get the same discriminator both times
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()
