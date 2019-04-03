from ops.basic_ops import downpool, conv, dense, mix_images, upscale

import tensorflow as tf


class GrowingNetwork:

    def __init__(self, channels, name, min_res=4, *args, **kwargs):
        self.channels = channels
        self.name = name
        self.min_res = min_res

    def get_res(self, block_number):
        return self.min_res * (2 ** block_number)

    def layer(self, t, n_channels, block_number=None, layer_number=None, **layer_params):
        return conv(t, n_channels, **layer_params)

    def dense_layer(self, t, n_channels, block_number=None, layer_number=None, **layer_params):
        return dense(t, n_channels, **layer_params)

    def block(self, t, n_channels, block_number=None, n_layers=2, reverse_layers=False, dense=False, **layer_params):
        layer_range = range(n_layers)
        if reverse_layers: layer_range = reversed(layer_range)

        for layer_number in layer_range:
            with tf.variable_scope('layer_{}'.format(layer_number)):

                if dense:
                    t = self.dense_layer(t, n_channels, block_number, layer_number, **layer_params)
                else:
                    t = self.layer(t, n_channels, block_number, layer_number, **layer_params)

        return t

    def get_train_op(self, loss, learning_rate, beta1, beta2):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
            with tf.control_dependencies(self.update_ops):
                train_op = optimizer.minimize(loss, var_list=list(self.vars.values()))
        return train_op

    def update_vars_and_ops(self):
        local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name[len(self.name):]: var for var in local_vars}
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)

    def network_fn(self, t_in, n_blocks, alpha):
        pass

    def __call__(self, t_in, n_blocks, alpha=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            t_out = self.network_fn(t_in, n_blocks, alpha)
            self.update_vars_and_ops()
            return t_out


class Generator(GrowingNetwork):

    def __init__(self, channels, name='Generator', min_res=4, *args, **kwargs):
        super().__init__(channels=channels, name=name, min_res=min_res, *args, **kwargs)

    def layer(self, g, n_channels, block_number=None, layer_number=None, **layer_params):

        if layer_number == 0:

            if block_number == 0:
                g = tf.expand_dims(g, 2)
                g = tf.expand_dims(g, 3)
                g = super().layer(g, n_channels, block_number, layer_number, mode='transpose', filter_size=self.min_res,
                    padding='VALID', output_shape=[tf.shape(g)[0], n_channels, self.min_res, self.min_res],
                    **layer_params)

            else:
                g = self.upsize(g)
                g = super().layer(g, n_channels, block_number, layer_number, **layer_params)

        else:
            g = super().layer(g, n_channels, block_number, layer_number, **layer_params)

        return g

    def to_rgb_block(self, g, up=False, **layer_params):
        x = conv(g, 3, filter_size=1, **layer_params)
        if up:
            x = self.rgb_upsize(x)
        return x

    def upsize(self, g):
        return upscale(g)

    def rgb_upsize(self, g):
        return upscale(g)

    def network_fn(self, z_in, n_blocks, alpha):
        g = z_in

        for block_number in range(n_blocks):
            res = self.get_res(block_number)
            n_channels = self.channels[block_number]

            with tf.variable_scope('block_{}'.format(res)):
                g = self.block(g, n_channels, block_number)

            if block_number == n_blocks - 2 and alpha is not None:
                with tf.variable_scope('to_rgb_block_{}'.format(res)):
                    x0 = self.to_rgb_block(g, up=True)

            if block_number == n_blocks - 1:
                with tf.variable_scope('to_rgb_block_{}'.format(res)):
                    x1 = self.to_rgb_block(g)

                if alpha is not None:
                    with tf.variable_scope('mix_images'):
                        x_out = mix_images(x0, x1, alpha)
                else:
                    x_out = x1

                self.update_vars_and_ops()
                return x_out


class Discriminator(GrowingNetwork):

    def __init__(self, channels, name='Discriminator', min_res=4, *args, **kwargs):
        super().__init__(channels=channels, name=name, min_res=min_res, *args, **kwargs)

    def layer(self, d, n_channels, block_number=None, layer_number=None, **layer_params):

        if layer_number == 0:

            if block_number == 0:
                d = super().layer(d, n_channels, filter_size=4, padding='VALID', **layer_params)
                d = tf.reshape(d, [-1, n_channels])

            else:
                d = super().layer(d, n_channels, block_number, layer_number, **layer_params)
                d = self.downsize(d)

        else:
            d = super().layer(d, n_channels, block_number, layer_number, **layer_params)

        return d

    def block(self, d, n_channels, block_number=None, n_layers=2, reverse_layers=True, dense=False, **layer_params):
        return super().block(d, n_channels, block_number, n_layers, reverse_layers, dense, **layer_params)

    def from_rgb_block(self, d, n_channels, down=False, **layer_params):
        if down:
            d = self.rgb_downsize(d)
        d = self.layer(d, n_channels, filter_size=1, **layer_params)
        return d

    def downsize(self, d):
        return downpool(d)

    def rgb_downsize(self, d):
        return downpool(d)

    def output_block(self, d):
        d = self.dense_layer(d, 1)
        return d

    def network_fn(self, x_in, n_blocks, alpha):
        n_channels = self.channels[min(n_blocks, len(self.channels) - 1)]

        for block_number in reversed(range(n_blocks)):
            res = self.get_res(block_number)

            if block_number == n_blocks - 1:
                with tf.variable_scope('from_rgb_block_{}'.format(res)):
                    d = self.from_rgb_block(x_in, n_channels)

            if block_number == n_blocks - 2 and alpha is not None:
                with tf.variable_scope('from_rgb_block_{}'.format(res)):
                    d0 = self.from_rgb_block(x_in, n_channels, down=True)
                    d1 = d

                with tf.variable_scope('mix_images'):
                    d = mix_images(d0, d1, alpha)

            n_channels = self.channels[block_number]

            with tf.variable_scope('block_{}'.format(res)):
                d = self.block(d, n_channels, block_number)

        with tf.variable_scope('output_block'):
            d = self.output_block(d)
            return d


class Encoder(Discriminator):

    def __init__(self, channels, output_channels, name='Encoder', min_res=4, *args, **kwargs):
        super().__init__(channels=channels, name=name, min_res=min_res, *args, **kwargs)
        self.output_channels = output_channels

    def output_block(self, d):
        layer_numbers = reversed(range(len(self.output_channels)))

        for layer_number, n_channels in zip(layer_numbers, self.output_channels):
            with tf.variable_scope('layer_{}'.format(layer_number)):
                d = self.dense_layer(d, n_channels, layer_number=layer_number)

        return d