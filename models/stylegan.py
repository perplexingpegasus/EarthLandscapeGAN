from config import stylegan_config
from models.networks import Discriminator, Generator
from models.progan import ProGanBuilder
from ops.advanced_ops import ADAin, minibatch_stddev, noise_input
from ops.basic_ops import add_b, blur, get_weights, leaky_relu, mix_images, tensor_to_images
from ops.loss_functions import non_saturating_loss, r1_penalty

import numpy as np
import tensorflow as tf


class StyleGanGenerator(Generator):

    def __init__(self, channels, name='Generator', min_res=4, mapping_layers=8, mapping_size=512, mapping_scale=0.01,
        *args, **kwargs):
        super().__init__(channels, name, min_res)

        self.mapping_layers = mapping_layers
        self.mapping_size = mapping_size
        self.mapping_scale = mapping_scale

    def layer(self, g, n_channels, block_number=None, layer_number=None, **layer_params):
        if layer_number == 0:

            if block_number == 0:
                const = get_weights([1, n_channels, self.min_res, self.min_res], 'learned_constant')
                g = tf.tile(const, [tf.shape(g)[0], 1, 1, 1])

            else:
                g = super().layer(g, n_channels, block_number, layer_number, bias=False, **layer_params)
                g = blur(g)

        else:
            g = super().layer(g, n_channels, block_number, layer_number, bias=False, **layer_params)

        g = noise_input(g, self.noise_imgs[block_number][layer_number])
        g = add_b(g)
        g = leaky_relu(g)
        g = ADAin(self.w[block_number], g)

        return g

    def dense_layer(self, z, n_channels, block_number=None, layer_number=None, **layer_params):
        z = super().dense_layer(z, n_channels, block_number, layer_number, **layer_params)
        z = leaky_relu(z)
        return z
    
    def mapping_block(self, z_in):
        return self.block(z_in, self.mapping_size, n_layers=self.mapping_layers, dense=True)

    def get_train_op(self, loss, learning_rate, beta1, beta2):
        network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)

        g_vars = []
        mn_vars = []
        for var in network_vars:
            if 'mapping_network' in var.name: mn_vars.append(var)
            else: g_vars.append(var)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name='Adam_G')
            mn_optimizer = tf.train.AdamOptimizer(learning_rate * self.mapping_scale, beta1, beta2, name='Adam_MN')

            gradients = tf.gradients(loss, g_vars + mn_vars)
            n_g_vars = len(g_vars)
            g_gradients = gradients[:n_g_vars]
            mn_gradients = gradients[n_g_vars:]

            with tf.control_dependencies(updates):
                g_train_op = g_optimizer.apply_gradients(zip(g_gradients, g_vars))
                mn_train_op = mn_optimizer.apply_gradients(zip(mn_gradients, mn_vars))
                train_op = tf.group(g_train_op, mn_train_op)

        return train_op

    def __call__(self, z_in, n_blocks, alpha=None, noise_imgs=None, psi=None, *args, **kwargs):

        if noise_imgs is not None:
            self.noise_imgs = noise_imgs
        else:
            self.noise_imgs = [[None] * 2] * n_blocks

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('mapping_network', reuse=tf.AUTO_REUSE):

                if z_in.get_shape().ndims == 3:
                    self._w = [self.mapping_block(z_in[:, i, :]) for i in range(n_blocks)]

                    if psi is not None:
                        with tf.variable_scope('truncation_trick'):

                            with tf.variable_scope('w_avg'):
                                self.w_avg = get_weights([self.mapping_size], 'w_avg', 'zeros', trainable=False)
                                new_w_avg = tf.reduce_mean(self._w[0], 0)
                                self.set_w_avg_op = self.w_avg.assign(new_w_avg)

                            self.w = [mix_images(self.w_avg, self._w[i], psi[i]) for i in range(n_blocks)]

                    else:
                        self.w = self._w

                else:
                    w = self.mapping_block(z_in)
                    self.w = [w] * n_blocks

            x_out = self.network_fn(z_in, n_blocks, alpha)
            return x_out


class StyleGanDiscriminator(Discriminator):

    def layer(self, d, n_channels, block_number=None, layer_number=None, **layer_params):

        if block_number == 0 and layer_number == 1:
            d = minibatch_stddev(d)

        if block_number != 0 and layer_number == 0:
            d = blur(d)

        d = super().layer(d, n_channels, block_number, layer_number, **layer_params)
        d = leaky_relu(d)

        return d

class StyleGanBuilder(ProGanBuilder):
    generator_cls = StyleGanGenerator
    discriminator_cls = StyleGanDiscriminator

    def __init__(self, logdir, arraydir=None,
        channels=stylegan_config['channels'],
        imgs_per_stage=stylegan_config['imgs_per_stage'],
        batch_size=stylegan_config['batch_size'],
        batch_repeats=stylegan_config['batch_repeats'],
        g_learning_rate=stylegan_config['g_learning_rate'],
        g_beta1=stylegan_config['g_beta1'],
        g_beta2=stylegan_config['g_beta2'],
        d_learning_rate=stylegan_config['d_learning_rate'],
        d_beta1=stylegan_config['d_beta1'],
        d_beta2=stylegan_config['d_beta2'],
        gp_lambda=stylegan_config['gp_lambda'],
        z_length=stylegan_config['z_length'],
        z_fixed_size=stylegan_config['z_fixed_size'],
        mapping_layers=stylegan_config['mapping_layers'],
        mapping_size=stylegan_config['mapping_size'],
        mapping_scale=stylegan_config['mapping_scale'],
        ema_decay=stylegan_config['ema_decay'],
        **model_params):

        super().__init__(logdir, arraydir, channels, imgs_per_stage, batch_size=batch_size, batch_repeats=batch_repeats,
            g_learning_rate=g_learning_rate, g_beta1=g_beta1, g_beta2=g_beta2, d_learning_rate=d_learning_rate,
            d_beta1=d_beta1, d_beta2=d_beta2, gp_lambda=gp_lambda, gp_gamma=None, z_length=z_length,
            z_fixed_size=z_fixed_size, mapping_layers=mapping_layers, mapping_size=mapping_size,
            mapping_scale=mapping_scale, ema_decay=ema_decay, **model_params)

    def initialize_networks(self):
        if self.trainer.res == 4:
            self.trainer.change_stage(8, False)

        super().initialize_networks()

    def build_placeholders(self):
        super().build_placeholders()

        n_blocks = self.trainer.n_blocks

        self.z_ema_placeholder = tf.placeholder(tf.float32, [None, n_blocks, self.trainer.z_length], 'z_ema')
        self.z_ema = self.z_ema_placeholder

        self.noise_img_placeholders = []
        self.psi_placeholders = []
        np.random.seed(0)

        for i in range(n_blocks):
            self.psi_placeholders.append(tf.placeholder_with_default(0.7, [], 'psi_{}'.format(i)))
            res = self.trainer.min_res * (2 ** i)
            shape = [None, 1, res, res]

            block_noise_imgs = []
            for j in range(2):
                imgs = np.random.normal(size=[5, *shape[1:]])
                imgs = np.tile(imgs, [6, 1, 1, 1])
                imgs = imgs.astype(np.float32)
                name = 'noise_imgs_{}_{}'.format(res, j)

                placeholder = tf.placeholder_with_default(imgs, shape, name)
                block_noise_imgs.append(placeholder)

            self.noise_img_placeholders.append(block_noise_imgs)

    def build_networks(self):
        n_blocks = self.trainer.n_blocks
        alpha = self.alpha if self.trainer.is_fading else None

        self.Gz = self.generator(self.z, n_blocks, alpha)
        self.Dz = self.discriminator(self.Gz, n_blocks, alpha)
        self.Dx = self.discriminator(self.x, n_blocks, alpha)

        if self.trainer.res < 512:
            self.EMA_Gz = self.ema_generator(self.z_ema, n_blocks, alpha,
                self.noise_img_placeholders, self.psi_placeholders)

        else:
            z_ema0, z_ema1 = self.z_ema[:15], self.z_ema[15:30]
            noise_img_placeholders0 = [
                [ph[:15] for ph in block_placeholders]
                for block_placeholders in self.noise_img_placeholders]
            noise_img_placeholders1 = [
                [ph[15:30] for ph in block_placeholders]
                for block_placeholders in self.noise_img_placeholders]

            EMA_Gz0 = self.ema_generator(z_ema0, n_blocks, alpha,
                noise_img_placeholders0, self.psi_placeholders)
            EMA_Gz1 = self.ema_generator(z_ema1, n_blocks, alpha,
                noise_img_placeholders1, self.psi_placeholders)
            self.EMA_Gz = tf.concat((EMA_Gz0, EMA_Gz1), 0)

    def build_loss_function(self):
        g_loss, d_loss = non_saturating_loss(self.Dz, self.Dx)
        gp = r1_penalty(self.Dx, self.x, self.trainer.gp_lambda)

        res = self.trainer.res
        with tf.variable_scope('loss_summaries_{}x{}'.format(res, res)):
            for loss, name in zip(
                (g_loss, d_loss, gp),
                ('generator_loss', 'discriminator_loss', 'gradient_penalty')
            ):
                self.scalars.append(loss)
                loss_sum = tf.summary.scalar(name, loss)
                self.scalar_sums.append(loss_sum)

        self.g_loss = g_loss
        self.d_loss = d_loss + gp

    def tile_z(self, z):
        z = np.expand_dims(z, 1)
        z = np.tile(z, [1, self.trainer.n_blocks, 1])
        return z

    @property
    def save_feed_dict(self):
        feed_dict = super().save_feed_dict
        feed_dict.update({
            self.z_ema_placeholder: self.tile_z(np.repeat(feed_dict[self.z_placeholder], 5, 0))
        })
        del feed_dict[self.z_placeholder]
        return feed_dict

    def run_save_ops(self):
        z = self.tile_z(self.trainer.z_batch(10000))
        self.sess.run(self.ema_generator.set_w_avg_op, {self.z_ema_placeholder: z})
        super().run_save_ops()

    def generate(self, z, psi=None, noise_imgs=None):
        feed_dict = {self.z_ema_placeholder: z}

        if psi is not None:
            for i in range(self.trainer.n_blocks):
                feed_dict.update({self.psi_placeholders[i]: psi[i]})

        if noise_imgs is not None:
            for i in range(self.trainer.n_blocks):
                for j in range(2):
                    feed_dict.update({self.noise_img_placeholders[i][j]: noise_imgs[i][j]})

        return self.sess.run(self.fake_imgs, feed_dict)


class StyleGanValidModel(StyleGanBuilder):

    def initialize_networks(self):
        self.ema_generator = self.generator_cls(self.trainer.channels, 'EMA_Generator', **self.trainer.model_params)

    def build_model(self):
        with tf.variable_scope('counters_and_placeholders'):
            self.build_counters_and_placeholders()

        self.build_networks()

        with tf.variable_scope('media_summaries'):
            self.build_media_summaries()

    def build_networks(self):
        n_blocks = self.trainer.n_blocks
        alpha = self.alpha if self.trainer.is_fading else None
        self.EMA_Gz = self.ema_generator(self.z_ema, n_blocks, alpha,
            self.noise_img_placeholders, self.psi_placeholders)

    def build_media_summaries(self):
        self.fake_imgs = tensor_to_images(self.EMA_Gz)