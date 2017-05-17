import tensorflow as tf
import numpy as np
import hypertools as hyp
import os
import argparse
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.vae import VAE, train_vae

# %%
    def test_mnist(n_epochs=50000,
                convolutional=False,
                variational=True,
                clustered=True,
                output_path="result_mnist_vae"):
        """Train an autoencoder on MNIST.

        This function will train an autoencoder on MNIST and also
        save many image files during the training process, demonstrating
        the latent space of the inner most dimension of the encoder,
        as well as reconstructions of the decoder.
        """
        # load MNIST
        n_code = 2
        n_clusters = 12
        mnist = MNIST(split=[0.8, 0.1, 0.1])
        ae = VAE(input_shape=[None, 784],
                 n_filters=[512, 256],
                 n_hidden=64,
                 n_code=n_code,
                 n_clusters=n_clusters,
                 activation=tf.nn.sigmoid,
                 convolutional=convolutional,
                 variational=variational,
                 clustered=clustered)

        n_examples = 8
        zs = np.random.uniform(
            -1.0, 1.0, [4, n_code]).astype(np.float32)
        zs = utils.make_latent_manifold(zs, n_examples)

        learning_rate = 0.02
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(ae['cost'])

        # We create a session to use the graph config = tf.ConfigProto()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # Fit all training data
        t_i = 0
        batch_i = 0
        batch_size = 200
        test_xs = mnist.test.images[:n_examples**2]
        utils.montage(test_xs.reshape((-1, 28, 28)), output_path + '/test_xs.png')
        # initial centers for Kmeans
        old_cent = sess.run(
            ae['z'], feed_dict={ae['x']: test_xs,
                                ae['train']: False,
                                ae['keep_prob']: 1.0})[:n_clusters]
        for epoch_i in range(n_epochs):
            train_i = 0
            train_cost = 0
            for batch_xs, _ in mnist.train.next_batch(batch_size):
                train_cost += sess.run([ae['cost'], optimizer], feed_dict={
                    ae['x']: batch_xs, ae['t']: batch_xs, ae['train']: True, ae['keep_prob']: 1.0,
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
                if batch_i % 1000 == 0:
                    # Plot example reconstructions from latent layer
                    recon = sess.run(
                        ae['y'], feed_dict={
                            ae['z']: zs,
                            ae['train']: False,
                            ae['keep_prob']: 1.0,
                            ae['old_cent']: old_cent})
                    m = utils.montage(recon.reshape((-1, 28, 28)),
                        output_path + '/manifold_%08d.png' % t_i)
                    m = utils.montage(recon.reshape((-1, 28, 28)),
                        output_path + '/manifold_latest.png')
                    # Plot example reconstructions
                    recon = sess.run(
                        ae['y'], feed_dict={ae['x']: test_xs[:n_examples**2],
                                            ae['train']: False,
                                            ae['keep_prob']: 1.0,
                                            ae['old_cent']: old_cent})
                    m = utils.montage(recon.reshape(
                        (-1, 28, 28)), output_path + '/reconstruction_%08d.png' % t_i)
                    m = utils.montage(recon.reshape(
                        (-1, 28, 28)), output_path + '/reconstruction_latest.png')
                    t_i += 1
                batch_i += 1

            valid_i = 0
            valid_cost = 0
            z_viz = []
            label_viz =[]
            for batch_xs, batch_ys in mnist.valid.next_batch(batch_size):
                valid_cost += sess.run([ae['cost']], feed_dict={
                    ae['x']: batch_xs, ae['t']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0,
                    ae['old_cent']: old_cent})[0]
                # Plot the latent variables
                z_viz = np.append(
                    z_viz,
                    sess.run([ae['z']], feed_dict={
                        ae['x']: batch_xs, ae['t']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0,
                        ae['old_cent']: old_cent})[0])
                label_viz = np.append(label_viz, batch_ys.argmax(1)).astype(int)
                valid_i += 1
            z_viz = np.reshape(z_viz, (-1, n_code))
            hyp.plot(z_viz, 'o', group=label_viz, show=False, save_path=output_path+'/scatter_%08d.png' % epoch_i)
            print('train:', train_cost / train_i, 'valid:', valid_cost / valid_i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument('-c',
        action="store_true",
        dest="convolutional", help='Whether use convolution or not')
    parser.add_argument('-v',
        action="store_true",
        dest="variational", help='Wether use latent variance or not')
    parser.add_argument('-k',
        action="store_true",
        dest="clustered", help='Whether use K-means or not')
    parser.add_argument('-o',
        action="store",
        dest="output_path",
        default="result_vae", help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    test_mnist(
        convolutional=results.convolutional,
        variational=results.variational,
        clustered=results.clustered,
        output_path=results.output_path)
