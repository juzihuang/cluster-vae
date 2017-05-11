import tensorflow as tf
import numpy as np
import os
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils

# %%
def test_celeb(n_epochs=50):
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
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./celeb.ckpt')


if __name__ == '__main__':
    test_celeb()
