import tensorflow as tf
c = tf.constant('TF version: {}'.format(tf.__version__))
with tf.Session() as sess:
    print(sess.run(c))

