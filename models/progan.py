from config import progan_config
from models.model_builder import ModelBuilder
from models.networks import Generator, Discriminator
from ops.advanced_ops import minibatch_stddev, pixelwise_norm
from ops.basic_ops import leaky_relu, mix_images, tensor_to_images
from ops.loss_functions import gradient_penalty, wgan_loss

from numpy import round
import tensorflow as tf


class ProGanGenerator(Generator):

    def layer(self, g, n_channels, block_number=None, layer_number=None, **layer_params):
        g = super().layer(g, n_channels, block_number, layer_number, **layer_params)
        g = leaky_relu(g)
        g = pixelwise_norm(g)
        return g


class ProGanDiscriminator(Discriminator):

    def layer(self, d, n_channels, block_number=None, layer_number=None, **layer_params):
        if block_number == 0 and layer_number == 1:
            d = minibatch_stddev(d)

        d = super().layer(d, n_channels, block_number, layer_number, **layer_params)
        d = leaky_relu(d)
        return d


class ProGanBuilder(ModelBuilder):
    generator_cls = ProGanGenerator
    discriminator_cls = ProGanDiscriminator

    def __init__(self, logdir, arraydir=None,
        channels=progan_config['channels'],
        imgs_per_stage=progan_config['imgs_per_stage'],
        batch_size=progan_config['batch_size'],
        batch_repeats=progan_config['batch_repeats'],
        g_learning_rate=progan_config['g_learning_rate'],
        g_beta1=progan_config['g_beta1'],
        g_beta2=progan_config['g_beta2'],
        d_learning_rate=progan_config['d_learning_rate'],
        d_beta1=progan_config['d_beta1'],
        d_beta2=progan_config['d_beta2'],
        gp_lambda=progan_config['gp_lambda'],
        gp_gamma=progan_config['gp_gamma'],
        z_length=progan_config['z_length'],
        ema_decay=progan_config['ema_decay'],
        **model_params):

        super().__init__(logdir, arraydir, channels, imgs_per_stage, batch_size=batch_size, batch_repeats=batch_repeats,
            g_learning_rate=g_learning_rate, g_beta1=g_beta1, g_beta2=g_beta2, d_learning_rate=d_learning_rate,
            d_beta1=d_beta1, d_beta2=d_beta2, gp_lambda=gp_lambda, gp_gamma=gp_gamma, z_length=z_length,
            ema_decay=ema_decay, **model_params)

    def initialize_networks(self):
        self.generator = self.generator_cls(self.trainer.channels, **self.trainer.model_params)
        self.ema_generator = self.generator_cls(self.trainer.channels, 'EMA_Generator', **self.trainer.model_params)
        self.discriminator = self.discriminator_cls(self.trainer.channels, **self.trainer.model_params)

    def build_model(self):
        super().build_model()
        with tf.variable_scope('exponential_moving_average'):
            self.build_ema_op()

    def build_networks(self):
        n_blocks = self.trainer.n_blocks
        alpha = self.alpha if self.trainer.is_fading else None

        self.Gz = self.generator(self.z, n_blocks, alpha)
        self.Dz = self.discriminator(self.Gz, n_blocks, alpha)
        self.Dx = self.discriminator(self.x, n_blocks, alpha)

        if self.trainer.res < 512:
            self.EMA_Gz = self.ema_generator(self.z, n_blocks, alpha)
        else:
            EMA_Gz0 = self.ema_generator(self.z[:15], n_blocks, alpha)
            EMA_Gz1 = self.ema_generator(self.z[15:30], n_blocks, alpha)
            self.EMA_Gz = tf.concat((EMA_Gz0, EMA_Gz1), 0)

        with tf.variable_scope('gradient_penalty_mix_images'):
            batch_size = tf.shape(self.x)[0]
            epsilon = tf.random_uniform([batch_size, 1, 1, 1], 0.0, 1.0)
            self.zx = mix_images(self.x, self.Gz, epsilon)
        self.Dzx = self.discriminator(self.zx, n_blocks, alpha)

    def build_ema_op(self):
        ema_assign_ops = []

        for var_name, var in self.generator.vars.items():
            if var.trainable:
                ema_var = self.ema_generator.vars[var_name]
                new_avg = mix_images(var, ema_var, self.trainer.ema_decay)
                ema_assign_ops.append(ema_var.assign(new_avg))

        ema_op = tf.group(*ema_assign_ops)
        self.train_ops.append(ema_op)

    def initialize_new_vars(self):
        global_vars = tf.global_variables()
        is_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        uninitialized_vars = [var for var, init in zip(global_vars, is_initialized) if not init]
        self.sess.run(tf.variables_initializer(uninitialized_vars))

        name_length = len(self.ema_generator.name)
        local_var_names = [
            var.name[name_length:] for var in uninitialized_vars
            if var.trainable and var.name.startswith(self.ema_generator.name)]

        with tf.variable_scope('copy_generator_vars'):
            copy_var_ops = [
                self.ema_generator.vars[name].assign(self.generator.vars[name])
                for name in local_var_names]

        self.sess.run(copy_var_ops)

    def build_loss_function(self):
        g_loss, d_loss = wgan_loss(self.Dz, self.Dx)
        gp = gradient_penalty(self.Dzx, self.zx, self.trainer.gp_lambda, self.trainer.gp_gamma)

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

    def build_optimizers(self):
        self.g_train = self.generator.get_train_op(
            self.g_loss, self.trainer.g_learning_rate, self.trainer.g_beta1, self.trainer.g_beta2)
        self.d_train = self.discriminator.get_train_op(
            self.d_loss, self.trainer.d_learning_rate, self.trainer.d_beta1, self.trainer.d_beta2)
        self.d_train = tf.group(self.d_train, self.increment_op)
        self.train_ops = [self.d_train, self.g_train]

    def build_media_summaries(self):
        self.fake_imgs = tensor_to_images(self.EMA_Gz)
        real_imgs = tensor_to_images(self.x)

        img_res = self.trainer.res
        if img_res < 128:
            img_res = 128
            self.fake_imgs = tf.image.resize_nearest_neighbor(self.fake_imgs, [img_res, img_res])
            real_imgs = tf.image.resize_nearest_neighbor(real_imgs, [img_res, img_res])

        fake_img_cols = [tf.concat([self.fake_imgs[j] for j in range(i * 5, (i + 1) * 5)], 0) for i in range(6)]
        fake_img_grid = tf.concat(fake_img_cols, 1)
        real_img_col = tf.concat([real_imgs[i] for i in range(5)], 0)
        black_bar = tf.zeros([img_res * 5, img_res // 3, 3], dtype=tf.uint8)

        big_img = tf.concat((fake_img_grid, black_bar, real_img_col), 1)
        big_img = tf.expand_dims(big_img, 0)

        img_sum = tf.summary.image(
            'image_preview_{}x{}{}'.format(
                self.trainer.res, self.trainer.res,
                '_fade' if self.trainer.is_fading else ''
            ), big_img)

        self.media_sums.append(img_sum)

    @property
    def save_feed_dict(self):
        return {self.z_placeholder: self.trainer.z_fixed, self.x_placeholder: self.trainer.x_batch(5)}

    def run_train_ops(self):
        feed_dict = self.train_feed_dict
        for _ in range(self.trainer.batch_repeats):
            for op in self.train_ops:
                self.sess.run(op, feed_dict)

    def run_summary_ops(self, start_time, start_img_step, cur_img_step):
        total_imgs, img_step, scalars, time_remaining = super().run_summary_ops(
            start_time, start_img_step, cur_img_step)

        g_loss, d_loss, gp = scalars
        high_res = self.trainer.res
        title_str = '{}x{}'.format(high_res, high_res)
        if self.trainer.is_fading:
            low_res = high_res // 2
            title_str = '{}x{} -> {}'.format(low_res, low_res, title_str)

        print('-------------------------------------------------------------------------------------')
        print(title_str)
        print('{}/{} images in current stage'.format(img_step, self.trainer.imgs_per_stage))
        print('{} total images'.format(total_imgs))
        print('generator loss: {}'.format(g_loss))
        print('discriminator loss: {}'.format(d_loss))
        print('gradient penalty: {}'.format(gp))
        print('time remaining: {}'.format(time_remaining))
        print('-------------------------------------------------------------------------------------')
        print('\n')

    def run_save_ops(self):
        super().run_save_ops()
        print('saved model and created image preview')
        print('\n')