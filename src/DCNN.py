import tensorflow as tf
import tensorflow.layers as layers

class DCNN:
    """
    This class implements a DCNN from https://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf
    """

    def __init__(self, pretrained=False, learning_rate=1e-3, train_switch_num=10000):
        """
        Sets up the model for training and inference
        :param pretrained: Whether or not to use pretrained weights
        :param learning_rate: The learning rate for training
        :param train_switch_num: The number of training examples after which the
            network will switch from traingin the sub-networks individually to
            training the overall network
        """

        self.num_train_exs_seen = 0
        self.train_switch_num = train_switch_num

        # Printout
        if pretrained:
            print("Building saved DCNN model...")
        else:
            print("Building new DCNN model...")

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
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])


        #Build Deconvolution CNN (2 hidden layers + an output)
        conv1 = layers.conv2d(self.input_placeholder, 38, (1, 121), 1, "same", activation=tf.nn.tanh, name="dcnn1")

        conv2 = layers.conv2d(conv1, 38, (121, 1), 1, "same", activation=tf.nn.tanh, name="dcnn2")

        conv3 = layers.conv2d(conv2, 1, (1, 1), 1, "same", activation=None, name="dcnn3")

        self.intermediate_representation = conv3

        #Build Outlier-rejection Deconvolution CNN (using outputs of DCNN as inputs)
        conv4 = layers.conv2d(self.intermediate_representation, 512, (16, 16), 1, "same", activation=tf.nn.tanh, name="odcnn1")

        conv5 = layers.conv2d(conv4, 512, (1, 1), 1, "same", activation=tf.nn.tanh, name="odcnn2")

        conv6 = layers.conv2d(conv5, 1, (8,8), 1, "same", activation=None, name="odcnn3")

        self.output = conv6

        dcnn_vars, odcnn_vars = [], []

        for dcnn_conv_layer in [conv1, conv2, conv3]:
            dcnn_vars.extend(dcnn_conv_layer.variables)

        for odcnn_conv_layer in [conv4, conv5, conv6]:
            odcnn_vars.extend(odcnn_conv_layer.variables)

        # Load pre-trained weights
        if pretrained:
            raise NotImplementedError()

        else:
            # If we aren't using pre-trained weights then we can just skip this part
            self.sess.run(tf.global_variables_initializer())
            self.loss = tf.losses.mean_squared_error(self.labels, self.output)
            self.dcnn_loss = tf.losses.mean_squared_error(self.labels, self.intermediate_representation)
            self.dcnn_optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.dcnn_train_op = self.dcnn_optimizer.minimize(self.dcnn_loss, var_list=dcnn_vars)

            self.odcnn_optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.odcnn_train_op = self.odcnn_optimizer.minimize(self.loss, var_list=odcnn_vars)

            self.overall_optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.overall_train_op = self.overall_optimizer.minimize(self.loss)

        # Init the variables relevant to the optimizers
        for optimizer in [self.dcnn_optimizer, self.odcnn_optimizer, self.overall_optimizer]:
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

        if self.num_train_exs_seen <= self.train_switch_num:
            dcnn_loss_value, _ = self.sess.run([self.dcnn_loss, self.dcnn_train_op], feed_dict=feed_dict)
            loss_value, _ = self.sess.run((self.loss, self.odcnn_train_op), feed_dict=feed_dict)
        else:
            loss_value, _ = self.sess.run((self.loss, self.overall_train_op), feed_dict=feed_dict)

        self.num_train_exs_seen += len(inputs)
        return loss_value

    def save_model(self):
        """
        Saves the model in the checkpoints folder
        :return: None
        """
        print("Saving model...")
        saver = tf.train.Saver()
        saver.save(self.sess, "./checkpoints/DCNNmodel.ckpt")
