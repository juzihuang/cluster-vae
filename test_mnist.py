# Those 3 lines must be setted when we ssh other servers without displaying
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.vae import VAE, train_vae

# %%
def test_mnist(n_epochs=5000,
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
    n_clusters = 10
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
    rng = numpy.random.RandomState(1)
    zs = rng.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    learning_rate = 0.02
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the graph config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(output_path + '/logs', sess.graph)

    # Fit all training data
    t_i = 0
    batch_i = 0
    batch_size = 512
    test_xs = mnist.test.images[:n_examples**2]
    utils.montage(test_xs.reshape((-1, 28, 28)), output_path + '/test_xs.png')

    for epoch_i in range(n_epochs):
        train_i = 0
        train_cost = 0
        for batch_xs, batch_ys in mnist.train.next_batch(batch_size):
            summary, cost_batch, _ = sess.run([ae['merged'], ae['cost'], optimizer], feed_dict={
                ae['x']: batch_xs, ae['t']: batch_xs,
                ae['train']: True,
                ae['keep_prob']: 1.0})
            train_cost += cost_batch

            # Get new centroids
            nebula = sess.run(
                ae['nebula'], feed_dict={
                    ae['x']: batch_xs,
                    ae['train']: False,
                    ae['keep_prob']: 1.0})

            train_i += 1
            if batch_i % 100 == 0:
                train_writer.add_summary(summary,
                    epoch_i*(mnist.train.images.shape[0]/batch_size) + train_i)
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'], feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0})
                #m = utils.montage(recon.reshape((-1, 28, 28)),
                #    output_path + '/manifold_%08d.png' % t_i)
                m = utils.montage(recon.reshape((-1, 28, 28)),
                    output_path + '/manifold_latest.png')
                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs[:n_examples**2],
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0})
                #m = utils.montage(recon.reshape(
                #    (-1, 28, 28)), output_path + '/reconstruction_%08d.png' % t_i)
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
                ae['x']: batch_xs, ae['t']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0})[0]
            # Plot the latent variables
            z_viz = np.append(
                z_viz,
                sess.run([ae['z']], feed_dict={
                    ae['x']: batch_xs, ae['t']: batch_xs, ae['train']: False, ae['keep_prob']: 1.0})[0])
            label_viz = np.append(label_viz, batch_ys.argmax(1)).astype(int)
            valid_i += 1
        z_viz = np.reshape(z_viz, (-1, n_code))

        print('train:', train_cost / train_i, 'valid:', valid_cost / valid_i)

        # Start ploting distributions on latent space
        with sns.color_palette(palette="hls", n_colors=n_clusters):
            g = sns.jointplot(z_viz[:,0], z_viz[:,1],
                xlim={-2.5,2.5}, ylim={-2.5,2.5},
                kind="kde", size=6, space=0.2, color='b')
            g.savefig(output_path+'/latent_distribution_latest.png')
            g = sns.jointplot(nebula[:,0], nebula[:,1],
                xlim={-2.5,2.5}, ylim={-2.5,2.5},
                size=6, space=0.2, color="r")
            g.savefig(output_path+'/centers_latest.png')
            pdd = pd.DataFrame(data=z_viz, index=label_viz, columns = ['x', 'y'])
            g = sns.JointGrid("x", "y", space=0.2,
                xlim={-2.5,2.5}, ylim={-2.5,2.5}, data=pdd)
            for idx, perviz in pdd.groupby(label_viz):
                sns.kdeplot(perviz["x"], ax=g.ax_marg_x, legend=False)
                sns.kdeplot(perviz["y"], ax=g.ax_marg_y, vertical=True, legend=False)
                g.ax_joint.plot(perviz["x"], perviz["y"], "o", ms=5)
            g.savefig(output_path+'/scatter_latest.png')

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
