import tensorflow as tf
import numpy as np
import os
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.vae import VAE, train_vae

# %%
def test_mnist(n_epochs=100):
    """Train an autoencoder on MNIST.

    This function will train an autoencoder on MNIST and also
    save many image files during the training process, demonstrating
    the latent space of the inner most dimension of the encoder,
    as well as reconstructions of the decoder.
    """
    output_path = "result_mnist_vae"
    # load MNIST
    n_code = 2
    n_clusters = 12
    mnist = MNIST(split=[0.8, 0.1, 0.1])
    # initial centers for Kmeans
    old_cent = np.random.uniform(
        -1.0, 1.0, [n_clusters, n_code]).astype(np.float32)
    # End
    ae = VAE(input_shape=[None, 784], n_filters=[512, 256],
             n_hidden=64, n_code=n_code, n_clusters=n_clusters,
             activation=tf.nn.sigmoid,
             convolutional=True,
             variational=True,
             clustered=False)

    n_examples = 100
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    learning_rate = 0.02
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the graph config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    t_i = 0
    batch_i = 0
    batch_size = 200
    test_xs = mnist.test.images[:n_examples]
    utils.montage(test_xs.reshape((-1, 28, 28)), output_path + '/test_xs.png')
    for epoch_i in range(n_epochs):
        train_i = 0
        train_cost = 0
        for batch_xs, _ in mnist.train.next_batch(batch_size):
            train_cost += sess.run([ae['cost'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['train']: True, ae['keep_prob']: 1.0,
                ae['old_cent']: old_cent})[0]

            # Get new centroids
            old_cent = sess.run(
                ae['new_cent'], feed_dict={ae['x']: test_xs,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['old_cent']: old_cent})
            # To fix: I don't know why there are nan in cent
            old_cent = np.nan_to_num(old_cent)

            train_i += 1
            if batch_i % 10 == 0:
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'], feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0,
                        ae['old_cent']: old_cent})
                m = utils.montage(recon.reshape((-1, 28, 28)),
                    output_path + '/manifold_%08d.png' % t_i)
                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['old_cent']: old_cent})
                m = utils.montage(recon.reshape(
                    (-1, 28, 28)), output_path + '/reconstruction_%08d.png' % t_i)
                t_i += 1
            batch_i += 1

        valid_i = 0
        valid_cost = 0
        for batch_xs, _ in mnist.valid.next_batch(batch_size):
            valid_cost += sess.run([ae['cost']], feed_dict={
                ae['x']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0,
                ae['old_cent']: old_cent})[0]
            valid_i += 1
        print('train:', train_cost / train_i, 'valid:', valid_cost / valid_i)

if __name__ == '__main__':
    test_mnist()
