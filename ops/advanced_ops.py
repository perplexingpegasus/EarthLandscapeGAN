from ops.basic_ops import conv, get_weights

import tensorflow as tf


def self_attention(x, key_channels, initialization='spectral_norm'):
    with tf.variable_scope('self_attention'):

        channels = x.shape[1].value
        height = tf.shape(x)[2]
        width = tf.shape(x)[3]

        with tf.variable_scope('f'):
            f = conv(x, key_channels, filter_size=1, initialization=initialization)
            f = tf.reshape(f, [-1, key_channels, height * width])

        with tf.variable_scope('g'):
            g = conv(x, key_channels, filter_size=1, initialization=initialization)
            g = tf.reshape(g, [-1, key_channels, height * width])

        with tf.variable_scope('h'):
            h = conv(x, channels, filter_size=1, initialization=initialization)
            h = tf.reshape(h, [-1, channels, height * width])


        s = tf.matmul(f, g, transpose_a=True)
        beta = tf.nn.softmax(s, axis=2)
        o = tf.matmul(h, beta)
        o = tf.reshape(o, [-1, channels, height, width])
        gamma = get_weights([1], 'gamma', 'zeros')

        y = x + o * gamma
        return y

def minibatch_stddev(x):
    with tf.variable_scope('minibatch_discrimination'):
        shape = tf.shape(x)
        group_size = tf.minimum(4, shape[0])
        y = tf.reshape(x, [group_size, -1, shape[1], shape[2], shape[3]])

        mu, sigma = tf.nn.moments(y, 0)
        sigma_avg = tf.reduce_mean(sigma, axis=[1, 2, 3], keepdims=True)
        sigma_avg = tf.tile(sigma_avg, [group_size, 1, shape[2], shape[3]])

        return tf.concat((x, sigma_avg), axis=1)

def pixelwise_norm(x):
    with tf.variable_scope('pixelwise_normalization'):
        pixel_var = tf.reduce_mean(tf.square(x), 1, keepdims=True)
        pixel_norm = x / tf.sqrt(pixel_var + 1e-8)
    return pixel_norm

def ADAin(w, x):
    with tf.variable_scope('ADAin'):
        w_length = int(w.get_shape()[1])
        image_channels = int(x.get_shape()[1])

        with tf.variable_scope('affine_transformation'):
            W = get_weights([w_length, image_channels * 2], 'W')
            b_ys = get_weights([image_channels], 'b_ys', 'ones')
            b_yb = get_weights([image_channels], 'b_yb', 'zeros')

            b = tf.concat((b_ys, b_yb), 0)
            y = tf.matmul(w, W) + b

        y = tf.expand_dims(y, 2)
        y = tf.expand_dims(y, 3)
        ys = y[:, :image_channels]
        yb = y[:, image_channels:]

        x_mu, x_sigma = tf.nn.moments(x, (2, 3), keep_dims=True)
        x_norm = (x - x_mu) / (x_sigma + 1e-8)

        a = ys * x_norm + yb
        return a

def noise_input(x, noise_img=None):
    with tf.variable_scope('noise_input'):
        batch_size = tf.shape(x)[0]
        channels = int(x.get_shape()[1])

        if noise_img is None:
            h = int(x.get_shape()[2])
            w = int(x.get_shape()[3])
            noise_img = tf.random_normal([batch_size, 1, h, w])

        scaling = get_weights([1, channels, 1, 1], 'noise_scaling', 'zeros')
        scaled_noise_imgs = noise_img * scaling

        y = x + scaled_noise_imgs
        return y