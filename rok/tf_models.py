"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import numpy as np
import tensorflow as tf

from constants import Constants as C
from utils import get_activation_fn


class BaseModel(object):
    """
    Base class that defines some functions and variables commonly used by all models. Subclass `BaseModel` to
    create your own models (cf. `DummyModel` for an example).
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.config = config  # The config parameters from the train.py script.
        self.data_placeholders = data_pl  # Placeholders where the input data is stored.
        self.mode = mode  # Train or eval.
        self.reuse = reuse  # If we want to reuse existing weights or not.
        self.source_seq_len = config["source_seq_len"]  # Length of the input seed.
        self.target_seq_len = config["target_seq_len"]  # Length of the predictions to be made.
        self.batch_size = config["batch_size"]  # Batch size.
        self.activation_fn_out = get_activation_fn(config["activation_fn"])  # Output activation function.
        self.data_inputs = data_pl[C.BATCH_INPUT]  # Tensor of shape (batch_size, seed length + target length)
        self.data_targets = data_pl[C.BATCH_TARGET]  # Tensor of shape (batch_size, seed length + target length)
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]  # Tensor of shape (batch_size, )
        self.data_ids = data_pl[C.BATCH_ID]  # Tensor of shape (batch_size, )
        self.is_eval = self.mode == C.EVAL  # If we are in evaluation mode.
        self.is_training = self.mode == C.TRAIN  # If we are in training mode.
        self.global_step = tf.train.get_global_step(graph=None)  # Stores the number of training iterations.

        print("data_inputs\t", self.data_inputs.get_shape())
        print("data_targets\t", self.data_targets.get_shape())
        print("data_seq_len\t", self.data_seq_len.get_shape(), self.data_seq_len)
        print("data_ids\t", self.data_ids.get_shape(), self.data_ids)

        # The following members should be set by the child class.
        self.outputs = None  # The final predictions.
        self.prediction_targets = None  # The targets.
        self.prediction_inputs = None  # The inputs used to make predictions.
        self.prediction_representation = None  # Intermediate representations.
        self.loss = None  # Loss op to be used during training.
        self.learning_rate = config["learning_rate"]  # Learning rate.
        self.parameter_update = None  # The training op.
        self.summary_update = None  # Summary op.

        # Hard-coded parameters that define the input size.
        self.JOINT_SIZE = 3*3
        self.NUM_JOINTS = 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

    def build_graph(self):
        """Build this model, i.e. its computational graph."""
        self.build_network()

    def build_network(self):
        """Build the core part of the model. This must be implemented by the child class."""
        raise NotImplementedError()

    def build_loss(self):
        """Build the loss function."""
        if self.is_eval:
            # In evaluation mode (for the validation set) we only want to know the loss on the target sequence,
            # because the seed sequence was just used to warm up the model.
            predictions_pose = self.outputs[:, -self.target_seq_len:, :]
            targets_pose = self.prediction_targets[:, -self.target_seq_len:, :]
        else:
            predictions_pose = self.outputs
            targets_pose = self.prediction_targets

        # Use MSE loss.
        with tf.name_scope("loss"):
            diff = targets_pose - predictions_pose
            self.loss = tf.reduce_mean(tf.square(diff))

    def optimization_routines(self):
        """Add an optimizer."""
        # Use a simple SGD optimizer.
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            # In case you want to do anything to the gradients, here you could do it.
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=self.global_step)

    def build_output_layer(self):
        """Build the final dense output layer without any activation."""
        with tf.variable_scope("output_layer", reuse=self.reuse):
            self.outputs = tf.layers.dense(self.prediction_representation, self.input_size,
                                           self.activation_fn_out, reuse=self.reuse)

            print("outputs\t", self.outputs.get_shape())

    def summary_routines(self):
        """Create the summary operations necessary to write logs into tensorboard."""
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to the summary name if needed.
        tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/learning_rate",
                              self.learning_rate,
                              collections=[self.mode + "/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")

    def step(self, session):
        """
        Perform one training step, i.e. compute the predictions when we can assume ground-truth is available.
        """
        raise NotImplementedError()

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available."""
        raise NotImplementedError()

    def predict(self, session):
        """
        Compute the predictions given the seed sequence without having access to the ground-truth values.
        """
        raise NotImplementedError()


class DummyModel(BaseModel):
    """
    A dummy RNN model.
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(DummyModel, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Extract some config parameters specific to this model
        self.cell_type = self.config["cell_type"]
        self.cell_size = self.config["cell_size"]
        self.input_hidden_size = self.config.get("input_hidden_size")

        # Prepare some members that need to be set when creating the graph.
        self.cell = None  # The recurrent cell. Defined in build_cell.
        self.initial_states = None  # The intial states of the RNN. Defined in build_network.
        self.rnn_outputs = None  # The outputs of the RNN layer.
        self.rnn_state = None  # The final state of the RNN layer.
        self.inputs_hidden = None  # The inputs to the recurrent cell.

        # How many steps we must predict.
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :-1, :]  # Pose input.
        self.prediction_targets = self.data_inputs[:, 1:, :]  # The target poses for every time step.
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

        print("prediction_inputs\t", self.prediction_inputs.get_shape())
        print("prediction_targets\t", self.prediction_targets.get_shape())
        print("prediction_seq_len\t", self.prediction_seq_len.get_shape())

        # Sometimes the batch size is available at compile time.
        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            # Sometimes it isn't. Use the dynamic shape instead.
            self.tf_batch_size = tf.shape(self.prediction_inputs)[0]

    def build_input_layer(self):
        """
        Here we can do some stuff on the inputs before passing them to the recurrent cell. The processed inputs should
        be stored in `self.inputs_hidden`.
        """
        # We could e.g. pass them through a dense layer
        if self.input_hidden_size is not None:
            with tf.variable_scope("input_layer", reuse=self.reuse):
                self.inputs_hidden = tf.layers.dense(self.prediction_inputs, self.input_hidden_size,
                                                     tf.nn.relu, reuse=self.reuse)
        else:
            self.inputs_hidden = self.prediction_inputs

        print("inputs_hidden:\t", self.inputs_hidden.get_shape())

    def build_cell(self):
        """Create recurrent cell."""
        with tf.variable_scope("rnn_cell", reuse=self.reuse):
            if self.cell_type == C.LSTM:
                cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
            elif self.cell_type == C.GRU:
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
            else:
                raise ValueError("Cell type '{}' unknown".format(self.cell_type))

            self.cell = cell

    def build_network(self):
        """Build the core part of the model."""
        self.build_input_layer()
        self.build_cell()

        self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                                 self.inputs_hidden,
                                                                 sequence_length=self.prediction_seq_len,
                                                                 initial_state=self.initial_states,
                                                                 dtype=tf.float32)
            self.prediction_representation = self.rnn_outputs
        self.build_output_layer()
        self.build_loss()

    def build_loss(self):
        super(DummyModel, self).build_loss()

    def step(self, session):
        """
        Run a training or validation step of the model.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
        """
        if self.is_training:
            # Training step.
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update,
                           self.data_inputs,
                           self.data_targets,
                           self.data_seq_len,
                           self.data_ids,
                           self.prediction_inputs,
                           self.prediction_targets,
                           self.inputs_hidden,
                           self.rnn_state,
                           self.rnn_outputs,
                           self.prediction_representation,
                           self.outputs,
                           self.global_step
                           ]
            outputs = session.run(output_feed)

            if outputs[15] < 3:
                print("\n")
                print("data_inputs", outputs[4].shape)
                print("data_targets", outputs[5].shape)
                print("data_seq_len", outputs[6].shape)
                print("data_ids", outputs[7].shape)
                print("prediction_inputs", outputs[8].shape)
                print("prediction_targets", outputs[9].shape)
                print("inputs_hidden", outputs[10].shape)
                print("rnn_state", outputs[11][0].shape, outputs[11][1].shape)
                print("rnn_outputs", outputs[12].shape)
                print("prediction_representation", outputs[13].shape)
                print("outputs", outputs[14].shape)

            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step (no backprop).
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available.
        Args:
          session: Tensorflow session object.
        Returns:
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets, seed sequence and
          unique sample IDs.
        """
        assert self.is_eval, "Only works in evaluation mode."

        # Get the current batch.
        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        targets = data_sample[:, self.source_seq_len:]

        seed_sequence = data_sample[:, :self.source_seq_len]
        predictions = self.sample(session, seed_sequence, prediction_steps=self.target_seq_len)

        return predictions, targets, seed_sequence, data_id

    def predict(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This assumes no
        ground-truth data is available.
        Args:
            session: Tensorflow session object.

        Returns:
            Prediction with shape (batch_size, self.target_seq_len, feature_size), seed sequence and unique sample IDs.
        """
        # `sampled_step` is written such that it works when no ground-truth data is available, too.
        predictions, _, seed, data_id = self.sampled_step(session)
        return predictions, seed, data_id

    def sample(self, session, seed_sequence, prediction_steps):
        """
        Generates `prediction_steps` may poses given a seed sequence.
        Args:
            session: Tensorflow session object.
            seed_sequence: A tensor of shape (batch_size, seq_len, feature_size)
            prediction_steps: How many frames to predict into the future.
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        # Now create predictions step-by-step.
        prediction = prediction[:, -1:]
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)
