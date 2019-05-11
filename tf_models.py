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
from utils import geodesic_distance
from motion_metrics import get_closest_rotmat
from utils import eulers_to_rotmats


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
        self.activation_fn_in = get_activation_fn(config["activation_input"])  # Input activation function.
        self.data_inputs = data_pl[C.BATCH_INPUT]  # Tensor of shape (batch_size, seed length + target length)
        self.data_targets = data_pl[C.BATCH_TARGET]  # Tensor of shape (batch_size, seed length + target length)
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]  # Tensor of shape (batch_size, )
        self.data_ids = data_pl[C.BATCH_ID]  # Tensor of shape (batch_size, )
        self.is_eval = self.mode == C.EVAL  # If we are in evaluation mode.
        self.is_training = self.mode == C.TRAIN  # If we are in training mode.
        self.global_step = tf.train.get_global_step(graph=None)  # Stores the number of training iterations.

        self.optimizer = self.config["optimizer"]
        self.loss = self.config["loss"]
        self.max_gradient_norm = 5.0

        # dense layers
        self.decoder_input_dense = None
        self.decoder_output_dense = None

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
        # Feature representation
        self.to_angles = self.config["to_angles"]
        if not self.to_angles:
            self.JOINT_SIZE = 3*3
            self.NUM_JOINTS = 15
            self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
            self.input_size = self.HUMAN_SIZE
        else:
            self.JOINT_SIZE = 3
            self.NUM_JOINTS = 15
            self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
            self.input_size = self.HUMAN_SIZE

        print("input_size", self.input_size)

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

        with tf.name_scope("loss"):
            if not self.to_angles:
                if self.loss == "geo":
                    # Geodesic loss
                    self.loss = geodesic_distance(targets_pose, predictions_pose)
                else:
                    # MSE
                    diff = targets_pose - predictions_pose
                    self.loss = tf.reduce_mean(tf.square(diff))
            else:
                diff = targets_pose - predictions_pose
                self.loss = tf.reduce_mean(tf.square(diff))

    def optimization_routines(self):
        """Add an optimizer."""
        # Use a simple SGD optimizer.
        if self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()

            gradients = tf.gradients(self.loss, params)
            # In case you want to do anything to the gradients, here you could do it.
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params),
                                                              global_step=self.global_step)

    def build_output_layer(self):
        """Build the final dense output layer without any activation."""
        with tf.variable_scope("output_layer", reuse=self.reuse):
            self.decoder_output_dense = tf.layers.Dense(self.input_size, use_bias=True,
                                                        activation=self.activation_fn_out)

            self.outputs = self.decoder_output_dense(self.prediction_representation)

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

        print("data_inputs\t", self.data_inputs.get_shape())
        print("data_targets\t", self.data_targets.get_shape())
        print("data_seq_len\t", self.data_seq_len.get_shape(), self.data_seq_len)
        print("data_ids\t", self.data_ids.get_shape(), self.data_ids)

        # How many steps we must predict.
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :-1, :]  # Pose input.
        self.prediction_targets = self.data_inputs[:, 1:, :]  # The target poses for every time step.
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

        print("source_seq_len", self.source_seq_len)
        print("target_seq_len", self.target_seq_len)
        print("sequence_len", self.sequence_length)
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
                           self.global_step,
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
                print("predictions_pose", outputs[14][:, -self.target_seq_len:, :].shape)
                print("targets_pose", outputs[9][:, -self.target_seq_len:, :].shape)

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

        if self.to_angles:
            targets = eulers_to_rotmats(targets)
            predictions = eulers_to_rotmats(predictions)

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
        seed_seq_len = np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: seed_seq_len}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        # print("\n\tsample")
        # print("\tone_step_seq_len", one_step_seq_len.shape)
        # print("\tseed_seq_len", seed_sequence.shape)
        # print("\tstate", len(state), state[0].shape, state[1].shape)
        # print("\tprediction", prediction.shape)

        # Now create predictions step-by-step.
        prediction = prediction[:, -1:]  # Last prediction from seed sequence
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)


class ZeroVelocityModel(BaseModel):
    """
    Zero Velocity Model. Baseline model for short-term human motion prediction.
    Every frame in predicted/output sequence is equal to the last input frame.
    """

    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(ZeroVelocityModel, self).__init__(config, data_pl, mode, reuse, **kwargs)

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
        self.inputs_hidden = tf.constant([0])

    def build_cell(self):
        """Create recurrent cell."""
        self.cell = tf.constant([0])

    def build_output_layer(self):
        """Build the final dense output layer without any activation."""
        self.outputs = tf.constant([0])

    def build_network(self):
        """Build the core part of the model."""
        self.build_input_layer()

        self.initial_states = tf.constant([0])
        self.rnn_outputs = tf.constant([0])
        self.rnn_state = tf.constant([0])

        self.build_output_layer()
        self.build_loss()

    def build_loss(self):
        self.loss = tf.constant(0)
        print("loss", self.loss.get_shape())

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
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step (no backprop).
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def optimization_routines(self):
        """Add an optimizer."""
        self.parameter_update = tf.constant([0])

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

        if self.to_angles:
            predictions = eulers_to_rotmats(predictions)

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
        last_frame = seed_sequence[:, -1, :]  # (16, 135)

        predictions = np.zeros(shape=(last_frame.shape[0], prediction_steps, last_frame.shape[1]))  # (16, 24, 135)
        for step in range(prediction_steps):
            predictions[:, step, :] = last_frame

        return predictions


class Seq2seq(BaseModel):
    """
    Seq2seq model.
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(Seq2seq, self).__init__(config, data_pl, mode, reuse, **kwargs)

        print(self.config)

        # Extract some config parameters specific to this model
        self.cell_type = self.config["cell_type"]
        self.cell_size = self.config["cell_size"]
        self.input_hidden_size = self.config.get("input_hidden_size")
        self.residuals = self.config["residuals"]
        self.sampling_loss = self.config["sampling_loss"]
        self.fidelity = self.config["fidelity"]
        self.continuity = self.config["continuity"]
        self.lambda_ = self.config["lambda_"]

        # Prepare some members that need to be set when creating the graph.
        self.cell = None  # The recurrent cell. (encoder)
        self.cell_decoder = None # The decoder cell.
        self.initial_states = None  # The intial states of the RNN.
        self.initial_states_decoder = None # The decoder initial state.
        self.rnn_outputs = None  # The outputs of the RNN layer.
        self.rnn_state = None  # The final state of the RNN layer.
        self.rnn_state_decoder = None
        self.inputs_hidden_encoder = None # The inputs to the encoder
        self.inputs_hidden = None  # The inputs to the decoder

        self.decoder_linear_output = None  # output linear dense layer; needed in case of sampling loss
        self.output_decoder_sampl_loss = None  # for more efficient predicting in case sampling loss

        # Fidelity discriminator
        self.fidelity_linear = None # Fidelity linear layer reference
        self.inputs_hidden_fid_tar = None
        self.inputs_hidden_fid_pred = None
        self.cell_fidelity = None
        self.fidelity_linear_out = None
        self.outputs_fid_tar = None
        self.outputs_fid_pred = None
        self.initial_states_fidelity = None
        self.state_fid_pred = None
        self.state_fid_tar = None
        self.loss_fidelity = None

        # continuity discriminator
        self.continuity_linear = None # continuity linear layer reference
        self.inputs_hidden_con_tar = None
        self.inputs_hidden_con_pred = None
        self.cell_continuity = None
        self.continuity_linear_out = None
        self.outputs_con_tar = None
        self.outputs_con_pred = None
        self.initial_states_continuity = None
        self.state_con_pred = None
        self.state_con_tar = None
        self.loss_continuity = None
        self.parameter_update_disc = None

        # How many steps we must predict.
        self.sequence_length = self.target_seq_len

        self.inputs_encoder = self.data_inputs[:, :self.source_seq_len-1, :]  # 0:119 -> 119 frames (seed without last)

        if not self.sampling_loss:
            self.prediction_inputs = self.data_inputs[:, self.source_seq_len-1:-1, :]  # 119:143 -> 24 frames (without last)
        else:
            self.prediction_inputs = self.data_inputs[:, self.source_seq_len-1, :]  # (16, 135) (last seed frame)

        self.prediction_targets = self.data_inputs[:, self.source_seq_len:, :]  # 120:144 -> 24 frames (last)

        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]),
                                          dtype=tf.int32)*self.sequence_length  # [24, ..., 24]
        self.prediction_seq_len_encoder = tf.ones((tf.shape(self.inputs_encoder)[0]),
                                                  dtype=tf.int32)*(self.source_seq_len-1)  # [119, ..., 119]

        # Sometimes the batch size is available at compile time.
        self.tf_batch_size = self.inputs_encoder.shape.as_list()[0]
        if self.tf_batch_size is None:
            # Sometimes it isn't. Use the dynamic shape instead.
            self.tf_batch_size = tf.shape(self.inputs_encoder)[0]

    def build_input_layer(self):
        """
        Here we can do some stuff on the inputs before passing them to the recurrent cell. The processed inputs should
        be stored in `self.inputs_hidden`.
        """
        # We could e.g. pass them through a dense layer
        if self.input_hidden_size is not None:
            if not self.sampling_loss:
                with tf.variable_scope("input_layer", reuse=self.reuse):
                    self.inputs_hidden = tf.layers.dense(self.prediction_inputs, self.input_hidden_size,
                                                         activation=self.activation_fn_in, reuse=self.reuse)

            with tf.variable_scope("input_layer_encoder", reuse=self.reuse):
                self.inputs_hidden_encoder = tf.layers.dense(self.inputs_encoder, self.input_hidden_size,
                                                             activation=self.activation_fn_in, reuse=self.reuse)
        else:
            self.inputs_hidden = self.prediction_inputs
            self.inputs_hidden_encoder = self.inputs_encoder

    def build_cell(self):
        """Create recurrent cell."""
        with tf.variable_scope("rnn_cell", reuse=self.reuse):
            if self.cell_type == C.LSTM:
                cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
                cell_decoder = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
                cell_fidelity = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
                cell_continuity = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
            elif self.cell_type == C.GRU:
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
                cell_decoder = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
                cell_fidelity = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
                cell_continuity = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
            else:
                raise ValueError("Cell type '{}' unknown".format(self.cell_type))

            self.cell = cell
            self.cell_decoder = cell_decoder
            self.cell_fidelity = cell_fidelity
            self.cell_continuity = cell_continuity

    def build_fidelity_input(self):
        """Fidelity linear input layer."""
        if self.input_hidden_size is not None:
            with tf.variable_scope("input_fidelity", reuse=self.reuse):
                self.fidelity_linear = tf.layers.Dense(self.input_hidden_size, use_bias=True,
                                                       activation=self.activation_fn_in)

                self.inputs_hidden_fid_tar = self.fidelity_linear(self.prediction_targets)
                self.inputs_hidden_fid_pred = self.fidelity_linear(
                    self.outputs)  # (16, 24, 135) -> # (16, 24, input_hidden_size)
        else:
            self.inputs_hidden_fid_tar = self.prediction_targets
            self.inputs_hidden_fid_pred = self.outputs

    def build_fidelity_output(self):
        """Linear layer for fidelity output."""
        with tf.variable_scope("fidelity_output", reuse=self.reuse):
            self.fidelity_linear_out = tf.layers.Dense(1, use_bias=True, activation=tf.nn.sigmoid)

            if self.cell_type == "gru":
                self.outputs_fid_tar = self.fidelity_linear_out(self.state_fid_tar)
                self.outputs_fid_pred = self.fidelity_linear_out(self.state_fid_pred)
            else:
                self.outputs_fid_tar = self.fidelity_linear_out(self.state_fid_tar[1])
                self.outputs_fid_pred = self.fidelity_linear_out(self.state_fid_pred[1])

    def build_loss_fidelity(self):
        self.loss_fidelity = tf.reduce_mean(tf.log(self.outputs_fid_tar + 1e-12)) + \
                             tf.reduce_mean(tf.log(1 - self.outputs_fid_pred + 1e-12))

        # self.loss = self.loss - self.lambda_ * tf.reduce_mean(tf.log(self.outputs_fid_pred + 1e-12))
        self.loss = self.loss + self.lambda_ * tf.reduce_mean(tf.log(1 - self.outputs_fid_pred + 1e-12))

    def build_continuity_input(self):
        """continuity linear input layer."""
        if self.input_hidden_size is not None:
            with tf.variable_scope("input_continuity", reuse=self.reuse):
                self.continuity_linear = tf.layers.Dense(self.input_hidden_size, use_bias=True,
                                                         activation=self.activation_fn_in)

                self.inputs_hidden_con_tar = self.continuity_linear(self.data_inputs)
                self.inputs_hidden_con_pred = self.continuity_linear(
                    tf.concat([self.data_inputs[:, :self.source_seq_len, :], self.outputs], axis=1))
        else:
            self.inputs_hidden_con_tar = self.data_inputs
            self.inputs_hidden_con_pred = tf.concat([self.data_inputs[:, :self.source_seq_len, :], self.outputs], axis=1)

    def build_continuity_output(self):
        """Linear layer for continuity output."""
        with tf.variable_scope("continuity_output", reuse=self.reuse):
            self.continuity_linear_out = tf.layers.Dense(1, use_bias=True, activation=tf.nn.sigmoid)

            if self.cell_type == "gru":
                self.outputs_con_tar = self.continuity_linear_out(self.state_con_tar)
                self.outputs_con_pred = self.continuity_linear_out(self.state_con_pred)
            else:
                self.outputs_con_tar = self.continuity_linear_out(self.state_con_tar[1])
                self.outputs_con_pred = self.continuity_linear_out(self.state_con_pred[1])

    def build_loss_continuity(self):
        self.loss_continuity = tf.reduce_mean(tf.log(self.outputs_con_tar + 1e-12)) + \
                             tf.reduce_mean(tf.log(1 - self.outputs_con_pred + 1e-12))

        # self.loss = self.loss - self.lambda_ * tf.reduce_mean(tf.log(self.outputs_con_pred + 1e-12))
        self.loss = self.loss + self.lambda_ * tf.reduce_mean(tf.log(1 - self.outputs_con_pred + 1e-12))

    def optimization_routines(self):
        """Add an optimizer."""
        # Use a simple SGD optimizer.
        if self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()

            params_gen = [var for var in params if not "continuity" in var.name and not "fidelity" in var.name]
            gradients = tf.gradients(self.loss, params_gen)
            # In case you want to do anything to the gradients, here you could do it.
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params_gen),
                                                              global_step=self.global_step)

            if self.fidelity:
                params_disc = [var for var in params if "continuity" in var.name or "fidelity" in var.name]
                gradients_disc = tf.gradients(- (self.loss_fidelity + self.loss_continuity), params_disc)
                clipped_gradients_disc, _ = tf.clip_by_global_norm(gradients_disc, self.max_gradient_norm)
                self.parameter_update_disc = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients_disc, params_disc))

    def build_network(self):
        """Build the core part of the model."""
        self.build_input_layer()
        self.build_cell()

        # Zero states
        self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        self.initial_states_fidelity = self.cell_fidelity.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        self.initial_states_continuity = self.cell_continuity.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)

        # RNN
        with tf.variable_scope("rnn_encoder", reuse=self.reuse):
            _, self.rnn_state = tf.nn.dynamic_rnn(self.cell, self.inputs_hidden_encoder,
                                                  sequence_length=self.prediction_seq_len_encoder,
                                                  initial_state=self.initial_states,
                                                  dtype=tf.float32)

        with tf.variable_scope("rnn_decoder", reuse=self.reuse):
            self.initial_states_decoder = self.rnn_state

            if not self.sampling_loss:
                self.rnn_outputs, self.rnn_state_decoder = tf.nn.dynamic_rnn(self.cell_decoder,
                                                                             self.inputs_hidden,
                                                                             sequence_length=self.prediction_seq_len,
                                                                             initial_state=self.initial_states_decoder,
                                                                             dtype=tf.float32)
                self.prediction_representation = self.rnn_outputs

                self.build_output_layer()
                if self.residuals:
                    self.residuals_decoder()

            else:
                # # METOD
                # with tf.variable_scope("linear_decoder_sampl_loss_", reuse=self.reuse):
                #     self.decoder_linear_output = tf.layers.Dense(self.input_size, use_bias=True, activation=None)
                #
                # state = self.initial_states_decoder  # Tuple((16, cell_size), (16, cell_size)) encoder hidden state
                #
                # seed = self.prediction_inputs
                # self.rnn_outputs = []  # feed the 120th frame, seed
                # for t in range(self.sequence_length):
                #     seed_, state = self.cell_decoder(inputs=seed,
                #                                      state=state)
                #
                #     seed_ = self.decoder_linear_output(seed_)
                #     if self.residuals:
                #         seed_ = tf.add(seed_, seed)  # down-project and add residual
                #     self.output_decoder_sampl_loss = seed_
                #     self.rnn_outputs.append(seed_)
                #     seed = seed_
                #
                # self.rnn_state_decoder = state
                # self.rnn_outputs = tf.stack(self.rnn_outputs, axis=1)
                #
                # self.outputs = self.rnn_outputs

                # ROK
                self.decoder_output_dense = tf.layers.Dense(self.input_size, use_bias=True,
                                                            activation=self.activation_fn_out, name="input_layer")

                self.decoder_input_dense = tf.layers.Dense(self.input_hidden_size, use_bias=True,
                                                           activation=self.activation_fn_in, name="output_layer")

                state = self.initial_states_decoder
                seed = self.prediction_inputs

                self.rnn_outputs = []
                for t in range(self.sequence_length):
                    tmp = self.decoder_input_dense(seed)

                    # RNN step
                    seed_, state = self.cell_decoder(inputs=tmp, state=state)
                    seed_ = self.decoder_output_dense(seed_)

                    if self.residuals:
                        seed_ = tf.add(seed_, seed)

                    self.rnn_outputs.append(seed_)
                    seed = seed_

                self.rnn_state_decoder = state
                self.rnn_outputs = tf.stack(self.rnn_outputs, axis=1)

                self.outputs = self.rnn_outputs

        self.build_loss()

        if self.fidelity:
            self.build_fidelity_input()

            with tf.variable_scope("rnn_fidelity", reuse=self.reuse):
                _, self.state_fid_tar = tf.nn.dynamic_rnn(self.cell_fidelity,
                                                          self.inputs_hidden_fid_tar,
                                                          sequence_length=self.prediction_seq_len,
                                                          initial_state=self.initial_states_fidelity,
                                                          dtype=tf.float32)

                _, self.state_fid_pred = tf.nn.dynamic_rnn(self.cell_fidelity,
                                                           self.inputs_hidden_fid_pred,
                                                           sequence_length=self.prediction_seq_len,
                                                           initial_state=self.initial_states_fidelity,
                                                           dtype=tf.float32)
            self.build_fidelity_output()
            self.build_loss_fidelity()

        if self.continuity:
            self.build_continuity_input()

            with tf.variable_scope("rnn_continuity", reuse=self.reuse):
                _, self.state_con_tar = tf.nn.dynamic_rnn(self.cell_continuity,
                                                          self.inputs_hidden_con_tar,
                                                          sequence_length=self.source_seq_len + self.prediction_seq_len,
                                                          initial_state=self.initial_states_continuity,
                                                          dtype=tf.float32)

                _, self.state_con_pred = tf.nn.dynamic_rnn(self.cell_continuity,
                                                           self.inputs_hidden_con_pred,
                                                           sequence_length=self.source_seq_len + self.prediction_seq_len,
                                                           initial_state=self.initial_states_continuity,
                                                           dtype=tf.float32)
            self.build_continuity_output()
            self.build_loss_continuity()

    def residuals_decoder(self):
        self.outputs = tf.add(self.outputs, self.prediction_inputs)

    def build_loss(self):
        super(Seq2seq, self).build_loss()

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
            if not self.fidelity:
                output_feed = [self.loss,
                               self.summary_update,
                               self.outputs,
                               self.parameter_update,
                               ]

                outputs = session.run(output_feed)
                # print("loss", outputs[0])
                return outputs[0], outputs[1], outputs[2]
            else:
                # Update all
                output_feed = [self.loss,
                               self.summary_update,
                               self.outputs,
                               self.parameter_update_disc,
                               self.parameter_update,
                               self.loss_fidelity,
                               self.loss_continuity,
                               self.global_step,
                               ]
                outputs = session.run(output_feed)

                # if outputs[7] % 20:
                #     print("loss", outputs[0])
                #     print("loss_fidelity", outputs[5])
                #     print("loss_continuity", outputs[6])
                #     print("loss_predictor", outputs[0] - self.lambda_*(outputs[6] + outputs[5]))

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
        targets = data_sample[:, self.source_seq_len:]  # 120:144 -> 24 (last frames)

        seed_sequence = data_sample[:, :self.source_seq_len]  # 0:120 -> 120 (seed)
        predictions = self.sample(session, seed_sequence, prediction_steps=self.target_seq_len)

        if self.to_angles:
            predictions = eulers_to_rotmats(predictions)

        # Guarantee valid rotation matrices
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]
        pred_val = np.reshape(predictions, [-1, self.NUM_JOINTS, 3, 3])  # (64, 24, 135)
        predictions = get_closest_rotmat(pred_val)  # (1536, 15, 3, 3)
        predictions = np.reshape(predictions, [batch_size, seq_length, self.input_size])  # (64, 24, 135)

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
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        seed_sequence_encoder = seed_sequence[:, :-1, :]  # (16, 119, 135)
        seed_decoder = seed_sequence[:, -1, :]  # (16, 135)

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.inputs_encoder: seed_sequence_encoder,  # 119 frames
                     self.prediction_seq_len_encoder:
                         np.ones(seed_sequence_encoder.shape[0])*seed_sequence_encoder.shape[1]}  # [119, ..., 119]
        state = session.run(self.rnn_state, feed_dict=feed_dict)

        # Now create predictions step-by-step.
        if not self.sampling_loss:
            prediction = seed_decoder[:, np.newaxis, :]
            predictions = []

            for step in range(prediction_steps):
                # get the prediction
                feed_dict = {self.prediction_inputs: prediction,
                             self.initial_states_decoder: state,
                             self.prediction_seq_len: one_step_seq_len}
                state, prediction = session.run([self.rnn_state_decoder, self.outputs], feed_dict=feed_dict)

                predictions.append(prediction)

            predictions = np.stack(predictions, axis=1).reshape((seed_decoder.shape[0],
                                                                 self.sequence_length, self.input_size))

        else:
            prediction = seed_decoder  # (16, 135)
            predictions = np.zeros(shape=(seed_decoder.shape[0], prediction_steps, self.input_size))  # (16, 24, 135)

            for step in range(prediction_steps):
                    feed_dict = {self.prediction_inputs: prediction,
                                 self.initial_states_decoder: state}
                    state, prediction = session.run([self.rnn_state_decoder, self.rnn_outputs], feed_dict=feed_dict)
                    prediction = prediction[:, 0, :]
                    predictions[:, step, :] = prediction

        return predictions
