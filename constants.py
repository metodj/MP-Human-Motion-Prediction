"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""


class Constants(object):
    # To control randomness.
    SEED = 4313

    # Run modes.
    TRAIN = 'training'
    TEST = 'test'
    EVAL = 'validation'

    # Recurrent cells.
    LSTM = 'lstm'
    GRU = 'gru'

    # Data Batch.
    BATCH_SEQ_LEN = "seq_len"
    BATCH_INPUT = "inputs"
    BATCH_TARGET = "targets"
    BATCH_ID = "id"

    # Activation functions.
    RELU = 'relu'

    # Metrics.
    METRIC_TARGET_LENGTHS = [5, 10, 19, 24, 34, 60]  # @ 60 fps, in ms: 83.3, 166.7, 316.7, 400, 566.7, 1000
