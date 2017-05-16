import tensorflow as tf
import numpy as np
import os
import libs
import argparse
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs.vae import VAE, train_vae
# %%
def test_celeb(n_epochs=50):
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

    """Train an autoencoder on Celeb Net.
    """
    files = CELEB()
    train_vae(
        files=files,
        input_shape=[218, 178, 3],
        batch_size=100,
        n_epochs=n_epochs,
        crop_shape=[64, 64, 3],
        crop_factor=0.8,
        convolutional=convolutional,
        variational=variational,
        clustered=clustered,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./celeb.ckpt',
        output_path=output_path)


if __name__ == '__main__':
    test_celeb()
