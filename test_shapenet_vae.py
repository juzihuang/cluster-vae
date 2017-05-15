import tensorflow as tf
import numpy as np
import os
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.vae import VAE, train_vae

# %%
def test_shapenet():
    # Train an autoencoder on Synthetic data rendered from ShapeNet.

    train_vae(
        files="./list_annotated_shapenet.csv",
        input_shape=[116, 116, 3],
        batch_size=64,
        n_epochs=500,
        crop_shape=[112, 112, 3],
        crop_factor=1.0,
        convolutional=True,
        variational=True,
        clustered=False,
        n_filters=[100, 100, 100, 100, 100],
        n_hidden=250,
        n_code=72,
        dropout=True,
        filter_sizes=[3, 3, 3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./shapenet.ckpt',
        output_path="result_shapenet_vae",
        input_type='file_in_csv')


if __name__ == '__main__':
    test_shapenet()
