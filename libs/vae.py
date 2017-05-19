"""Clustered/Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Yida Wang, May 2017
"""
import tensorflow as tf
import numpy as np
import os
import csv
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils


def VAE(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        n_clusters = 10,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False,
        clustered=False):
    """(Variational) (Convolutional) (Denoising) (Clustered) Autoencoder.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    n_hidden : int, optional
        Only applied when variational=True.  This refers to the first fully
        connected layer prior to the variational embedding, directly after
        the encoding.  After the variational embedding, another fully connected
        layer is created with the same size prior to decoding.  Set to 0 to
        not use an additional hidden layer.
    n_code : int, optional
        Only applied when variational=True.  This refers to the number of
        latent Gaussians to sample for creating the inner most encoding.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.nn.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.
    denoising : bool, optional
        Whether or not to apply denoising.  If using denoising, you must feed a
        value for 'corrupt_prob', as returned in the dictionary.  1.0 means no
        corruption is used.  0.0 means every feature is corrupted.  Sensible
        values are between 0.5-0.8.
    convolutional : bool, optional
        Whether or not to use a convolutional network or else a fully connected
        network will be created.  This effects the n_filters parameter's
        meaning.
    variational : bool, optional
        Whether or not to create a variational embedding layer.  This will
        create a fully connected layer after the encoding, if `n_hidden` is
        greater than 0, then will create a multivariate gaussian sampling
        layer, then another fully connected layer.  The size of the fully
        connected layers are determined by `n_hidden`, and the size of the
        sampling layer is determined by `n_code`.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            't': Target Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'centroids': Centers of the latent spaces
            'keep_prob': Amount to keep when using Dropout
            'corrupt_prob': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    t = tf.placeholder(tf.float32, input_shape, 't')
    old_cent = tf.placeholder(tf.float32, [n_clusters, n_code], 'old_cent')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])

    # apply noise if denoising
    x_ = (utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)) if denoising else x

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x_) if convolutional else x_
    current_input = x_tensor

    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    dims = current_input.get_shape().as_list()
    flattened = utils.flatten(current_input)

    if n_hidden:
        h = utils.linear(flattened, n_hidden, name='W_fc')[0]
        h = activation(batch_norm(h, phase_train, 'fc/bn'))
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
    else:
        h = flattened

    # Sample from posterior
    with tf.variable_scope('variational'):
        z_mu = utils.linear(h, n_code, name='mu')[0]
        z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]
        tf.summary.histogram('z_mu', z_mu)
        tf.summary.histogram('z_log_sigma', z_log_sigma)
        if variational:
            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.stack([tf.shape(x)[0], n_code]))
            z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))
        else:
            z = z_mu
        tf.summary.histogram('z', z)
    if n_hidden:
        h = utils.linear(z, n_hidden, name='fc_t')[0]
        h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
        if dropout:
            h = tf.nn.dropout(h, keep_prob)
    else:
        h = z

    size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
    h = utils.linear(h, size, name='fc_t2')[0]
    current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
    if dropout:
        current_input = tf.nn.dropout(current_input, keep_prob)

    if convolutional:
        current_input = tf.reshape(
            current_input, tf.stack([
                tf.shape(current_input)[0],
                dims[1],
                dims[2],
                dims[3]]))

    with tf.variable_scope('clustered'):
        if clustered:
            # K-Means cluster assisted clustering for latent space
            # Here we can not use tf.Variable() to define centroids
            # centroids = tf.slice(tf.random_shuffle(z), [0, 0], [n_clusters, -1])
            points_expanded = tf.expand_dims(z, 0)
            centroids_expanded = tf.expand_dims(old_cent, 1)

            # This will be a distance matrix by utilizing TensorFlow substract in
            # "Broadcasting" manner which is similar to numpy:
            # z -> [1, N, n_code]
            # old_cent -> [n_clusters, 1, n_code]
            # distance -> [n_clusters, N, n_code]
            distances = tf.subtract(points_expanded, centroids_expanded)

            # distance -> [n_clusters, N]
            distances = tf.reduce_sum(tf.square(distances), 2)

            # Returns the index with the smallest value across axes of a tensor.
            assignments = tf.argmin(distances, 0)

            # Updating the centroids according to all labeled data
            means = []
            for c in range(n_clusters):
                means.append(tf.reduce_mean(
                  tf.gather(z,
                            tf.reshape(tf.where(tf.equal(assignments, c)),
                                       [1,-1])),
                  reduction_indices=[1]))

            new_cent = tf.concat(means, 0)
        else:
            new_cent = old_cent
        ## KNN ending

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h

    y = current_input
    t_flat = utils.flatten(t)
    y_flat = utils.flatten(y)

    # l2 loss
    loss_t = tf.reduce_sum(tf.squared_difference(t_flat, y_flat), 1)
    tf.summary.scalar('cost_t', tf.reduce_mean(loss_t))

    if variational:
        # variational lower bound, kl-divergence
        loss_z = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma -
            tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)
        # add l2 loss
        cost = tf.reduce_mean(loss_t + loss_z)
        tf.summary.scalar('cost_z', tf.reduce_mean(loss_z))
    else:
        cost = tf.reduce_mean(loss_t)

    if clustered:
        # kmeans cluster loss optimization for latent space of vae
        #loss_c = tf.reduce_sum(tf.reduce_min(distances, axis=0))
        loss_c = tf.reduce_min(distances, axis=0)
        cost += tf.reduce_mean(loss_c)
        tf.summary.scalar('cost_c', tf.reduce_mean(loss_c))

    merged = tf.summary.merge_all()

    return {'cost': cost, 'Ws': Ws,
            'x': x, 't': t, 'z': z, 'y': y,
            'old_cent': old_cent,
            'new_cent': new_cent,
            'keep_prob': keep_prob,
            'corrupt_prob': corrupt_prob,
            'train': phase_train,
            'merged': merged}


def train_vae(files,
              input_shape,
              learning_rate=0.0001,
              batch_size=128,
              n_epochs=50,
              n_examples=8,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              n_clusters = 12,
              convolutional=True,
              variational=True,
              clustered=False,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              output_path="result",
              ckpt_name="vae.ckpt",
              input_type='file_list'):
    """General purpose training of a (Variational) (Convolutional) (Clustered)
        Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
        List of paths to images.
    input_shape : list
        Must define what the input image's shape is.
    learning_rate : float, optional
        Learning rate.
    batch_size : int, optional
        Batch size.
    n_epochs : int, optional
        Number of epochs.
    n_examples : int, optional
        Number of example to use while demonstrating the current training
        iteration's reconstruction.  Creates a square montage, so:
        n_examples**2 = 16, 25, 36, ... 100.
    crop_shape : list, optional
        Size to centrally crop the image to.
    crop_factor : float, optional
        Resize factor to apply before cropping.
    n_filters : list, optional
        Same as VAE's n_filters.
    n_hidden : int, optional
        Same as VAE's n_hidden.
    n_code : int, optional
        Same as VAE's n_code.
    convolutional : bool, optional
        Use convolution or not.
    variational : bool, optional
        Use variational layer or not.
    filter_sizes : list, optional
        Same as VAE's filter_sizes.
    dropout : bool, optional
        Use dropout or not
    keep_prob : float, optional
        Percent of keep for dropout.
    activation : function, optional
        Which activation function to use.
    img_step : int, optional
        How often to save training images showing the manifold and
        reconstruction.
    save_step : int, optional
        How often to save checkpoints.
    output_path : str, optional
        Defien a path for saving result and sample images
    ckpt_name : str, optional
        Checkpoints will be named as this, e.g. 'model.ckpt'
    input_type : str, Choose path or list for loading data
    """
    batch = create_input_pipeline(
        files=files,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape,
        input_type=input_type)

    ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             clustered=clustered,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             n_clusters=n_clusters,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the config = tf.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(output_path + '/logs', sess.graph)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(output_path + '/' + ckpt_name + '.index') or os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0

    # Get the number of training samples
    if input_type == 'file_in_csv':
        with open(files,"r") as f:
            reader = csv.reader(f,delimiter = ",")
            data = list(reader)
            n_files = len(data)
    elif input_type == 'file_list':
        n_files = len(files)

    if input_type == 'file_list':
        test_xs = sess.run(batch) / 255.0
        utils.montage(test_xs[:n_examples**2], output_path + '/test_xs.png')
    elif input_type == 'file_in_csv':
        test_xs, test_ts, _ = sess.run(batch)
        test_xs /= 255.0
        test_ts /= 255.0
        utils.montage(test_xs[:n_examples**2], output_path + '/test_xs.png')
        utils.montage(test_ts[:n_examples**2], output_path + '/test_ts.png')

    try:
        # initial centers for Kmeans
        old_cent = sess.run(
            ae['z'], feed_dict={ae['x']: test_xs,
                                ae['train']: False,
                                ae['keep_prob']: 1.0})[:n_clusters]
        # End
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            if input_type == 'file_list':
                batch_xs = sess.run(batch) / 255.0
                batch_ts = batch_xs
            elif input_type == 'file_in_csv':
                batch_xs, batch_ts, _ = sess.run(batch)
                batch_xs /= 255.0
                batch_ts /= 255.0
            summary, train_cost, _ = sess.run(
                [ae['merged'], ae['cost'], optimizer], feed_dict={
                    ae['x']: batch_xs, ae['t']: batch_ts, ae['train']: True,
                    ae['keep_prob']: keep_prob,
                    ae['old_cent']: old_cent})

            train_writer.add_summary(summary, epoch_i*(n_files/batch_size) + batch_i)
            print(batch_i, train_cost)

            # Get new centroids
            old_cent = sess.run(
                ae['new_cent'], feed_dict={ae['x']: batch_xs,
                                    ae['train']: False,
                                    ae['keep_prob']: 1.0,
                                    ae['old_cent']: old_cent})
            # To fix: I don't know why there are nan in cent
            old_cent = np.nan_to_num(old_cent)

            cost += train_cost
            if batch_i % (n_files/batch_size) == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                cost = 0
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'], feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0,
                        ae['old_cent']: old_cent})
                utils.montage(recon.reshape([-1] + crop_shape),
                    output_path + '/manifold_%08d.png' % t_i)
                utils.montage(recon.reshape([-1] + crop_shape),
                    output_path + '/manifold_latest.png')

                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs[:n_examples**2],
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0,
                                        ae['old_cent']: old_cent})
                print('reconstruction (min, max, mean):',
                    recon.min(), recon.max(), recon.mean())
                utils.montage(recon.reshape([-1] + crop_shape),
                              output_path+'/reconstruction_%08d.png' % t_i)
                utils.montage(recon.reshape([-1] + crop_shape),
                            output_path+'/reconstruction_latest.png')
                t_i += 1

            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver.save(sess, output_path + "/" + ckpt_name,
                           global_step=batch_i,
                           write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()
