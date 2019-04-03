import numpy as np
import tensorflow as tf


def get_weights(shape, name, initialization='runtime_scaling', transpose=False, lr_scale=1.0, trainable=True):

    def get_he_stddev(shape, transpose):
        if transpose and len(shape) == 4:
            fan_in = float(shape[0] * shape[1] * shape[3])
        else:
            fan_in = float(np.prod(shape[:-1]))
        return tf.sqrt(2.0 / fan_in)

    if initialization == 'zeros':
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    elif initialization == 'ones':
        initializer = tf.constant_initializer(1.0 / lr_scale, dtype=tf.float32)
    elif initialization == 'he':
        stddev = get_he_stddev(shape, transpose) / lr_scale
        initializer = tf.random_normal_initializer(stddev=stddev)
    else:
        stddev = 1.0 / lr_scale
        initializer = tf.random_normal_initializer(stddev=stddev)

    weights = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)

    if initialization == 'runtime_scaling':
        stddev = get_he_stddev(shape, transpose)
        weights = weights * (stddev * lr_scale)
    elif initialization == 'spectral_norm':
        weights = spectral_norm(weights, transpose)

    return weights

def spectral_norm(w, transpose=False):
    with tf.variable_scope('spectral_norm'):
        shape = tf.shape(w)

        if transpose:
            out_channels = shape[-2]
            w = tf.transpose(w, (0, 1, 3, 2))
        else:
            out_channels = shape[-1]

        w = tf.reshape(w, [-1, out_channels])
        u = tf.get_variable('u', [1, out_channels], initializer=tf.random_normal_initializer(), trainable=False)

        v_ = tf.matmul(u, w, transpose_b=True)

        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)
        sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)

        with tf.control_dependencies([tf.assign(u, u_hat)]):
            w = w / sigma
            if transpose:
                w = tf.transpose(w, (0, 1, 3, 2))
            w = tf.reshape(w, shape)

        return w

def add_b(x, initialization='zeros', lr_scale=1.0):

    with tf.variable_scope('bias'):

        channels = int(x.get_shape()[1])
        if len(x.get_shape()) == 2:
            b_shape = [channels]
        else:
            b_shape = [1, channels, 1, 1]

        b = get_weights(b_shape, 'b', initialization, lr_scale=lr_scale)
        y = x + b

    return y

def conv(x, out_channels, filter_size=3, k=1, padding='SAME', output_shape=None, mode=None,
    initialization='runtime_scaling', lr_scale=1.0, bias=True):

    with tf.variable_scope('conv{}'.format(('_' + mode) if mode is not None else '')):
        input_channels = int(x.get_shape()[1])

        if mode == 'transpose' or mode == 'up':
            transpose = True
            filter_shape = [filter_size, filter_size, out_channels, input_channels]
        else:
            transpose = False
            filter_shape = [filter_size, filter_size, input_channels, out_channels]

        W = get_weights(filter_shape, 'W', initialization, transpose, lr_scale)

        if mode == 'up':
            W = tf.pad(W, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            W = tf.add_n([W[1:, 1:], W[:-1, 1:], W[1:, :-1], W[:-1, :-1]])

            N = tf.shape(x)[0]
            h = int(x.get_shape()[2])
            w = int(x.get_shape()[3])
            output_shape = [N, out_channels, h * 2, w * 2]
            y = tf.nn.conv2d_transpose(x, W, output_shape, [1, 1, 2, 2], padding=padding, data_format='NCHW')

        elif mode == 'down':
            W = tf.pad(W, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
            W = tf.add_n([W[1:, 1:], W[:-1, 1:], W[1:, :-1], W[:-1, :-1]])
            W = W * 0.25
            y = tf.nn.conv2d(x, W, [1, 1, 2, 2], padding=padding, data_format='NCHW')

        elif mode == 'transpose':
            y = tf.nn.conv2d_transpose(x, W, output_shape, [1, 1, k, k], padding=padding, data_format='NCHW')

        else:
            y = tf.nn.conv2d(x, W, [1, 1, k, k], padding=padding, data_format='NCHW')

    if bias:
        y = add_b(y, lr_scale=lr_scale)

    return y

def dense(x, output_size, initialization='runtime_scaling', lr_scale=1.0, bias=True):

    with tf.variable_scope('dense'):
        input_channels = int(x.get_shape()[1])
        W = get_weights([input_channels, output_size], 'W', initialization, lr_scale=lr_scale)
        y = tf.matmul(x, W)

    if bias:
        y = add_b(y, lr_scale=lr_scale)

    return y

def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)

def batch_norm(x, is_training=True):
    return tf.contrib.layers.batch_norm(
        x, is_training=is_training, scale=True, epsilon=1e-5, data_format='NCHW', fused=True)

def mix_images(x0, x1, alpha):
    return (1.0 - alpha) * x0 + alpha * x1

def scale_uint8(x):
    x = tf.to_float(x)
    return (x / 127.5) - 1

def tensor_to_images(x):
    x = tf.transpose(x, (0, 2, 3, 1))
    x = tf.clip_by_value(x, -1.0, 1.0)
    x = (x + 1) * 127.5
    imgs = tf.cast(x, tf.uint8)
    return imgs

# resizing ops

def depthwise_conv(x, f, k=1):
    channels = int(x.get_shape()[1])

    f = np.reshape(f, [*f.shape, 1, 1])
    f = np.tile(f, [1, 1, channels, 1])
    f = tf.constant(f, name='filter', dtype=tf.float32)

    x = tf.nn.depthwise_conv2d(x, f, [1, 1, k, k], padding='SAME', data_format='NCHW')
    return x

def _blur(x):
    f = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125 , 0.25 , 0.125 ],
        [0.0625, 0.125, 0.0625]
    ], dtype=np.float32)
    return depthwise_conv(x, f)

def _downpool(x):
    f = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ], dtype=np.float32)
    return depthwise_conv(x, f, k=2)

def _upscale(x):
    channels = x.get_shape()[1]
    height = x.get_shape()[2]
    width = x.get_shape()[3]
    y = tf.reshape(x, [-1, channels, height, 1, width, 1])
    y = tf.tile(y, [1, 1, 1, 2, 1, 2])
    y = tf.reshape(y, [-1, channels, height * 2, width * 2])
    return y

def custom_grad_func(zeroth, first, second):
    @tf.custom_gradient
    def f0(x):
        y = zeroth(x)
        @tf.custom_gradient
        def f1(dy):
            dx = first(dy)
            def f2(ddx):
                return second(ddx)
            return dx, f2
        return y, f1
    return f0

def blur(x):
    with tf.variable_scope('blur'):
        return custom_grad_func(_blur, _blur, _blur)(x)

def downpool(x):
    def first_grad(x):
        return _upscale(x * 0.25)

    with tf.variable_scope('downpool'):
        return custom_grad_func(_downpool, first_grad, _downpool)(x)

def upscale(x):
    def first_grad(x):
        f = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ], dtype=np.float32)
        return depthwise_conv(x, f, k=2)

    with tf.variable_scope('upscale'):
        return custom_grad_func(_upscale, first_grad, _upscale)(x)

def resize_bilinear(x, new_res):
    with tf.variable_scope('resize_bilinear'):
        channels = int(x.get_shape()[1])
        cur_res = int(x.get_shape()[2])

        x = tf.reshape(x, (-1, cur_res, cur_res))
        x = tf.expand_dims(x, 3)

        y = tf.image.resize_bilinear(x, (new_res, new_res), align_corners=True)
        y = tf.squeeze(y, 3)
        y = tf.reshape(y, (-1, channels, new_res, new_res))
        return y

def scope(name):
    def outer(func):
        def inner(*args, **kwargs):
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        return inner
    return outer