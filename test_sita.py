import tensorflow as tf
import numpy as np
import os
import argparse
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils
from libs.vae import VAE, train_vae

# %%
def test_sita():
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

    """Train an autoencoder on Sita Sings The Blues.
    """
    if not os.path.exists('sita'):
        os.system('wget -c http://ossguy.com/sita/Sita_Sings_the_Blues_640x360_XviD.avi')
        os.mkdir('sita')
        os.system('ffmpeg -i Sita_Sings_the_Blues_640x360_XviD.avi -r 60 -f' +
                  ' image2 -s 160x90 sita/sita-%08d.jpg')
    files = [os.path.join('sita', f) for f in os.listdir('sita')]

    train_vae(
        files=files,
        input_shape=[90, 160, 3],
        batch_size=64,
        n_epochs=5000,
        crop_shape=[90, 160, 3],
        crop_factor=1.0,
        convolutional=results.convolutional,
        variational=results.variational,
        clustered=results.clustered,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.sigmoid,
        ckpt_name='./sita.ckpt',
        output_path=results.output_path)


if __name__ == '__main__':
    test_sita()
