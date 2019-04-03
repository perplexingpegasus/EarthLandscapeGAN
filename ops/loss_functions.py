import tensorflow as tf


def wgan_loss(Dz, Dx):
    with tf.variable_scope('generator_loss'):
        g_loss = tf.reduce_mean(-Dz)

    with tf.variable_scope('discriminator_loss'):
        d_loss = tf.reduce_mean(Dz - Dx)

    return g_loss, d_loss

def non_saturating_loss(Dz, Dx):
    fake_sce = lambda x: tf.maximum(x, 0) + tf.log(1 + tf.exp(-tf.abs(x)))
    real_sce = lambda x: fake_sce(x) - x

    with tf.variable_scope('generator_loss'):
        g_loss = tf.reduce_mean(real_sce(Dz))

    with tf.variable_scope('discriminator_loss'):
        d_loss = tf.reduce_mean(real_sce(Dx)) + tf.reduce_mean(fake_sce(Dz))

    return g_loss, d_loss

def gradient_penalty(Dzx, zx, gp_lambda=10.0, gp_gamma=1.0):
    with tf.variable_scope('gradient_penalty'):

        gradients = tf.gradients(tf.reduce_sum(Dzx), [zx])[0]
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), (1, 2, 3)) + 1e-8)
        gp = gp_lambda * tf.reduce_mean(tf.square(gradient_norm / gp_gamma - 1.0))

        return gp

def r1_penalty(Dx, x, gp_lambda=5.0):
    with tf.variable_scope('r1_penalty'):

        gradients = tf.gradients(tf.reduce_sum(Dx), [x])[0]
        gradient_norm = tf.reduce_sum(tf.square(gradients), (1, 2, 3))
        r1 = gp_lambda * tf.reduce_mean(gradient_norm)

        return r1