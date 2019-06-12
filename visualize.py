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
import subprocess
import shutil
import numpy as np
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.mplot3d import Axes3D

from motion_metrics import get_closest_rotmat
from motion_metrics import is_valid_rotmat


_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']


class Visualizer(object):
    """
    Helper class to visualize SMPL joint angles parameterized as rotation matrices.
    """
    def __init__(self, fk_engine, video_path=None):
        self.fk_engine = fk_engine
        self.video_path = video_path
        self.is_sparse = True
        self.expected_n_input_joints = len(self.fk_engine.major_joints) if self.is_sparse else self.fk_engine.n_joints

    def visualize_with_gt(self, seed, prediction, target, title):
        """
        Visualize prediction and ground truth side by side.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            target: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        self.visualize_rotmat(seed, prediction, target, title)

    def visualize(self, seed, prediction, title):
        """
        Visualize prediction only.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        self.visualize_rotmat(seed, prediction, title=title)

    def visualize_rotmat(self, seed, prediction, target=None, title=''):
        def _to_positions(angles_):
            full_seq = np.concatenate([seed, angles_], axis=0)

            # Make sure the rotations are valid.
            full_seq_val = np.reshape(full_seq, [-1, n_joints, 3, 3])
            full_seq = get_closest_rotmat(full_seq_val)
            full_seq = np.reshape(full_seq, [-1, n_joints * dof])

            # Check that rotation matrices are valid.
            full_are_valid = is_valid_rotmat(np.reshape(full_seq, [-1, n_joints, 3, 3]))
            assert full_are_valid, 'rotation matrices are not valid rotations'

            # Compute positions.
            if self.is_sparse:
                full_seq_pos = self.fk_engine.from_sparse(full_seq, return_sparse=False)  # (N, full_n_joints, 3)
            else:
                full_seq_pos = self.fk_engine.from_rotmat(full_seq)

            # Swap y and z because SMPL defines y to be up.
            full_seq_pos = full_seq_pos[..., [0, 2, 1]]
            return full_seq_pos

        assert seed.shape[-1] == prediction.shape[-1] == self.expected_n_input_joints * 9
        n_joints = self.expected_n_input_joints
        dof = 9

        pred_pos = _to_positions(prediction)
        positions = [pred_pos]
        colors = [_colors[0]]
        titles = ['prediction']

        if target is not None:
            assert prediction.shape[-1] == target.shape[-1]
            assert prediction.shape[0] == target.shape[0]
            targ_pos = _to_positions(target)
            positions.append(targ_pos)
            colors.append(_colors[1])
            titles.append('target')

        visualize_positions(positions=positions,
                            colors=colors,
                            titles=titles,
                            fig_title=title,
                            parents=self.fk_engine.parents,
                            change_color_after_frame=(seed.shape[0], None),
                            video_path=self.video_path)


def visualize_positions(positions, colors, titles, fig_title, parents, change_color_after_frame=None, overlay=False,
                        fps=60, video_path=None):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: A list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame.
        colors: List of color for each entry in `positions`.
        titles: List of titles for each entry in `positions`.
        fig_title: Title for the entire figure.
        parents: Skeleton structure.
        fps: Frames per second.
        change_color_after_frame: After this frame id, the color of the plot is changed (for each entry in `positions`).
        overlay: If true, all entries in `positions` are plotted into the same subplot.
        video_path: If not None, the animation is saved as a movie instead of shown interactively.
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]
        lines_j = [
            ax.plot(joints[0:1, n,  0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                    markersize=2.0, color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)
        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')
        # ax.axis('off')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=0, azim=-56)

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines, parents, colors):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and change_color_after_frame[l] and num >= change_color_after_frame[l]:
                    points_j[k].set_color(_colors[2])
                else:
                    points_j[k].set_color(colors[l])

                k += 1
        time_passed = '{:>.2f} seconds passed'.format(1/60.0*num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length,
                                       fargs=(pos, all_lines, parents, colors + [colors[0]]),
                                       interval=1000/fps)

    if video_path is None:
        # interactive
        plt.show()
    else:
        # save to video file
        print('saving video to {}'.format(video_path))
        save_animation(fig, seq_length, update_frame, [pos, all_lines, parents, colors + [colors[0]]],
                       video_path=video_path)
    plt.close()


def save_animation(fig, seq_length, update_func, update_func_args, video_path,
                   start_recording=0, end_recording=None, fps_in=60, fps_out=30):
    """
    Save animation as a video. This requires ffmpeg to be installed.
    Args:
        fig: Figure where animation is displayed.
        seq_length: Total length of the animation.
        update_func: Update function that is driving the animation.
        update_func_args: Arguments for `update_func`.
        video_path: Where to store the video.
        start_recording: Frame index where to start recording.
        end_recording: Frame index where to stop recording (defaults to `seq_length`, exclusive).
        fps_in: Input FPS.
        fps_out: Output FPS.
    """
    tmp_path = os.path.join(video_path, "tmp_frames")

    # Cleanup.
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)

    start_frame = start_recording
    end_frame = end_recording or seq_length

    # Save all the frames into the temp folder.
    for j in range(start_frame, end_frame):
        update_func(j, *update_func_args)
        fig.savefig(os.path.join(tmp_path, 'frame_{:0>4}.{}'.format(j, "png")), dip=1000)

    # Get unique movie name.
    counter = 0
    movie_name = os.path.join(video_path, "vid{}.avi".format(counter))
    while os.path.exists(movie_name):
        counter += 1
        movie_name = os.path.join(video_path, "vid{}.avi".format(counter))

    # Create the movie from the stored files using ffmpeg.
    command = ['ffmpeg',
               '-start_number', str(start_frame),
               '-framerate', str(fps_in),  # input framerate, must be this early, otherwise it is ignored
               '-r', str(fps_out),  # output framerate
               '-loglevel', 'panic',
               '-i', os.path.join(tmp_path, 'frame_%04d.png'),
               '-c:v', 'ffv1',
               '-pix_fmt', 'yuv420p',
               '-y',
               movie_name]
    FNULL = open(os.devnull, 'w')
    subprocess.Popen(command, stdout=FNULL).wait()
    FNULL.close()

    # Cleanup.
    shutil.rmtree(tmp_path, ignore_errors=True)


def _get_random_sample(tfrecords_path, n_samples=10):
    """Return `n_samples` many random samples from the tfrecords found unter `tfrecords_path`."""

    def _parse_tf_example(proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
        }

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])
        return parsed_features

    tf_data = tf.data.TFRecordDataset.list_files(tfrecords_path)
    tf_data = tf_data.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1)
    tf_data = tf_data.map(functools.partial(_parse_tf_example), 4)

    iterator = tf_data.make_one_shot_iterator()
    samples = []
    counter = 0
    for s in iterator:
        if counter >= n_samples:
            break
        samples.append(s["poses"].numpy())
        counter += 1

    return samples


if __name__ == '__main__':
    import tensorflow as tf
    import functools
    from fk import SMPLForwardKinematics

    # You will need a rather new version of TensorFlow to use eager execution.
    tf.enable_eager_execution()

    # Where the data is stored.
    # data_path = "./data/validation/poses-?????-of-?????"
    data_path = "/cluster/project/infk/hilliges/lectures/mp19/project4/validation/poses-?????-of-?????"

    # Get some random samples.
    samples = _get_random_sample(data_path, n_samples=5)

    # If we set the video path, the animations will be saved to video instead of shown interactively.
    video_path = os.environ['HOME'] if 'HOME' in os.environ else './'
    video_path = os.path.join(video_path, "videos")

    # Visualize each of them.
    visualizer = Visualizer(SMPLForwardKinematics(), video_path=video_path)
    for i, sample in enumerate(samples):
        visualizer.visualize(sample[:120], sample[120:], 'random validation sample {}'.format(i))
