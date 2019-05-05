"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import os
import glob
import json
import argparse
import numpy as np
import tensorflow as tf

# RS
import sys
import datetime
# RS

import tf_models as models
from tf_data import TFRecordMotionDataset
from constants import Constants as C
from utils import export_results
from utils import export_code
from visualize import Visualizer
from fk import SMPLForwardKinematics


def create_and_restore_test_model(session, experiment_dir, args):
    """
    Creates and restores the test model stored in the given directory.
    Args:
        session: The GPU session.
        experiment_dir: Where the model checkpoints and its config is stored.
        args: The commandline arguments of this script.

    Returns:
        The test model, the test data, and the config of the model.

    """
    config = json.load(open(os.path.abspath(os.path.join(experiment_dir, 'config.json')), 'r'))

    # Store seed and target sequence length in the config.
    # For the test set, these are hard-coded to be compatible with the submission requirements.
    config["target_seq_len"] = 24
    config["source_seq_len"] = 120

    # For the test data set, we don't have labels, so the window length is just the length of the seed.
    window_length = config["source_seq_len"]
    data_path = args.data_dir
    test_data_path = os.path.join(data_path, "test", "poses-?????-of-?????")
    meta_data_path = os.path.join(data_path, "training", "stats.npz")

    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          extract_random_windows=False,
                                          num_parallel_calls=16)
        test_pl = test_data.get_tf_samples()

    # Select the type of model we want to use.
    if config['model_type'] == "dummy":
        model_cls = models.DummyModel
    elif config["model_type"] == "zero_velocity":
        model_cls = models.ZeroVelocityModel
    elif config['model_type'] == 'seq2seq':
        model_cls = models.Seq2seq
    else:
        raise Exception("Unknown model type.")

    # Create the model.
    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.EVAL,
            reuse=False,
            dtype=tf.float32,
            is_test=True)
        test_model.build_graph()

    # Count number of trainable parameters.
    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    if not config["model_type"] == "zero_velocity":
        # Restore model parameters.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)

        # Restore the latest checkpoint found in `experiment_dir`.
        ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

        if ckpt and ckpt.model_checkpoint_path:
            # Check if the specific checkpoint exists
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("Loading model checkpoint {0}".format(ckpt_name))
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("could not load checkpoint")

    return test_model, test_data, config


def evaluate_model(sess, eval_model, eval_data):
    """
    Make a full pass on the test set and return the results.
    Args:
        sess: The active session.
        eval_model: The model we want to evaluate.
        eval_data: The data we want to evaluate.

    Returns:
        The results stored in a dictionary {"file_id" => (prediction, seed)}
    """
    # Initialize iterator.
    eval_iter = eval_data.get_iterator()
    sess.run(eval_iter.initializer)

    eval_result = dict()
    try:
        while True:
            # Get the predictions. Must call the function that works without having access to the ground-truth data,
            # as there is no ground-truth for the test set.
            prediction, seed_sequence, data_id = eval_model.predict(sess)

            # Store each test sample and corresponding predictions with the unique sample IDs.
            for i in range(prediction.shape[0]):
                eval_result[data_id[i].decode("utf-8")] = (prediction[i], seed_sequence[i])
    except tf.errors.OutOfRangeError:
        pass
    return eval_result


def evaluate(experiment_dir, args):
    """
    Evaluate the model stored in the given directory. It loads the latest available checkpoint and iterates over
    the test set.
    Args:
        experiment_dir: The model directory.
        args: Commandline arguments.
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        test_model, test_data, config = create_and_restore_test_model(sess, experiment_dir, args)

        print("Evaluating test set ...")
        eval_result = evaluate_model(sess, test_model, test_data)

        if args.export:
            # Export the results into a csv file that can be submitted.
            fname = os.path.join(experiment_dir, "predictions_in{}_out{}.csv".format(config['source_seq_len'],
                                                                                     config['target_seq_len']))
            export_results(eval_result, fname)

            # Export a zip file containing the code that generated the results.
            code_files = glob.glob('./*.py', recursive=False)
            export_code(code_files, os.path.join(experiment_dir, 'code.zip'))

        if args.visualize:
            # Visualize the seed sequence and the prediction for some random samples in the test set.
            fk_engine = SMPLForwardKinematics()
            visualizer = Visualizer(fk_engine)
            n_samples_viz = 10
            rng = np.random.RandomState(42)
            idxs = rng.randint(0, len(eval_result), size=n_samples_viz)
            sample_keys = [list(sorted(eval_result.keys()))[i] for i in idxs]
            for k in sample_keys:
                visualizer.visualize(eval_result[k][1], eval_result[k][0], title=k)


EXPERIMENT_TIMESTAMP = datetime.datetime.now().strftime("%d_%H-%M")
LOG_FILE = "./logs/log_" + EXPERIMENT_TIMESTAMP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str, default="./data", help='Where the data is stored.')
    parser.add_argument('--save_dir', required=True, type=str, default="./experiments", help='Where models are stored.')
    parser.add_argument('--model_id', required=True, type=str, help='Which model to load (its timestamp).')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--visualize', action="store_true", help='Visualize some model predictions.')
    parser.add_argument('--export', action="store_true", help="Export predictions to a csv file.")
    parser.add_argument("--log", action="store_true", help="create log file")

    args = parser.parse_args()

    if args.log:
        sys.stdout = open(LOG_FILE, "w")

    try:
        experiment_dir = glob.glob(os.path.join(args.save_dir, args.model_id + "-*"), recursive=False)[0]
    except IndexError:
        raise Exception("Model " + str(args.model_id) + " is not found in " + str(args.save_dir))

    evaluate(experiment_dir, args)
    sys.stdout.close()

