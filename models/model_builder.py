from data.model_trainer import ModelTrainer
from ops.basic_ops import _downpool, mix_images, scale_uint8, upscale

import datetime as dt
import os
import tensorflow as tf


class ModelBuilder:

    def __init__(self, logdir, arraydir=None, channels=None, imgs_per_stage=None, batch_size=None, **model_params):
        restored = ModelTrainer.is_saved(logdir)
        if restored:
            self.trainer = ModelTrainer.load(logdir)

        else:
            if arraydir is None:
                raise ValueError('"arraydir" must be provided if model is not being restored')
            if channels is None:
                raise ValueError('"channels" must be provided if model is not being restored')
            if imgs_per_stage is None:
                raise ValueError('"imgs_per_stage" must be provided if model is not being restored')
            if batch_size is None:
                raise ValueError('"batch_size" must be provided if model is not being restored')

            self.trainer = ModelTrainer(logdir, arraydir, channels, imgs_per_stage, batch_size, **model_params)

        self.initialize_networks()
        self.initialize_graph()
        if not restored:
            self.initialize_new_vars()
        else:
            self.restore_networks()

    def initialize_networks(self):
        pass

    def initialize_graph(self):
        tf.reset_default_graph()
        self.build_model()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(self.trainer.logdir, graph=self.sess.graph)
        self.saver = tf.train.Saver()

    def initialize_new_vars(self):
        global_vars = tf.global_variables()
        is_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        uninitialized_vars = [var for var, init in zip(global_vars, is_initialized) if not init]
        self.sess.run(tf.variables_initializer(uninitialized_vars))

    def restore_networks(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.trainer.logdir))

    def build_model(self):
        self.reset_op_lists()
        with tf.variable_scope('counters_and_placeholders'):
            self.build_counters_and_placeholders()

        self.build_networks()

        with tf.variable_scope('loss_function'):
            self.build_loss_function()
        with tf.variable_scope('optimizers'):
            self.build_optimizers()
        with tf.variable_scope('media_summaries'):
            self.build_media_summaries()

    def reset_op_lists(self):
        self.train_ops = []
        self.scalars = []
        self.scalar_sums = []
        self.media = []
        self.media_sums = []

    def build_counters_and_placeholders(self):
        with tf.variable_scope('counters'):
            self.build_counters()
        with tf.variable_scope('alpha'):
            self.build_alpha()
        with tf.variable_scope('placeholders'):
            self.build_placeholders()
        with tf.variable_scope('counter_ops'):
            self.build_counter_ops()

    def build_counters(self):
        self.total_imgs = tf.Variable(0, name='total_images', trainable=False, dtype=tf.int32)
        self.img_step = tf.Variable(0, name='image_step', trainable=False, dtype=tf.int32)

    def build_alpha(self):
        if self.trainer.is_fading:
            fade_in = tf.divide(tf.to_float(self.img_step), self.trainer.imgs_per_stage)
            self.alpha = tf.clip_by_value(fade_in, 0.0, 1.0)

    def build_placeholders(self):
        self.z_placeholder = tf.placeholder(tf.float32, [None, self.trainer.z_length], 'z')
        self.z = self.z_placeholder

        res = self.trainer.res
        self.x_placeholder = tf.placeholder(tf.uint8, [None, 3, res, res], 'x')
        x1 = scale_uint8(self.x_placeholder)

        if self.trainer.is_fading:
            x0 = upscale(_downpool(x1))
            self.x = mix_images(x0, x1, self.alpha)

        else:
            self.x = x1

    def build_counter_ops(self):
        batch_size = tf.shape(self.x)[0]

        inc_total_imgs = tf.assign_add(self.total_imgs, batch_size)
        inc_img_step = tf.assign_add(self.img_step, batch_size)
        self.increment_op = tf.group(inc_total_imgs, inc_img_step)

        self.reset_img_step = tf.assign(self.img_step, 0)

    def build_networks(self):
        pass

    def build_loss_function(self):
        pass

    def build_optimizers(self):
        pass

    def build_media_summaries(self):
        pass

    def update_network(self):
        self.change_stage()
        self.reset_optimizers()
        self.build_model()
        self.initialize_new_vars()
        self.reset_and_save_networks()

    def change_stage(self):
        self.sess.run(self.reset_img_step)
        self.trainer.change_stage()

    def reset_optimizers(self):
        optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'optimizers')
        self.sess.run(tf.variables_initializer(optimizer_vars))

    def reset_and_save_networks(self):
        self.saver = tf.train.Saver()
        self.save_model()

        del self.saver, self.sess, self.writer
        self.initialize_graph()
        self.restore_networks()

    def save_model(self):
        self.saver.save(
            self.sess, os.path.join(self.trainer.logdir, "model.ckpt"),
            global_step=self.total_imgs)
        self.trainer.save()

    @property
    def train_feed_dict(self):
        return {self.z_placeholder: self.trainer.z_batch(), self.x_placeholder: self.trainer.x_batch()}

    @property
    def summary_feed_dict(self):
        return self.train_feed_dict

    @property
    def save_feed_dict(self):
        return self.train_feed_dict

    def get_summaries(self, data, sums, feed_dict):
        total_imgs, img_step, *data_and_sums = self.sess.run(
            [self.total_imgs, self.img_step, *data, *sums], feed_dict,
            options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

        i = len(data)
        data, sums = data_and_sums[:i], data_and_sums[i:]

        for s in sums:
            self.writer.add_summary(s, total_imgs)

        return total_imgs, img_step, data

    def get_time_remaining(self, start_time, start_img_step, cur_img_step):
        time_per_img = (dt.datetime.now() - start_time) / (cur_img_step - start_img_step)
        time_remaining = (self.trainer.imgs_per_stage - cur_img_step) * time_per_img
        return time_remaining

    def run_train_ops(self):
        feed_dict = self.train_feed_dict
        for op in self.train_ops:
            self.sess.run(op, feed_dict)

    def run_summary_ops(self, start_time, start_img_step, cur_img_step):
        feed_dict = self.summary_feed_dict
        total_imgs, img_step, scalars = self.get_summaries(self.scalars, self.scalar_sums, feed_dict)
        time_remaining = self.get_time_remaining(start_time, start_img_step, cur_img_step)
        return total_imgs, img_step, scalars, time_remaining

    def run_save_ops(self):
        feed_dict = self.save_feed_dict
        self.save_model()
        total_imgs, img_step, media = self.get_summaries(self.media, self.media_sums, feed_dict)
        return total_imgs, img_step, media

    def train_model(self, summary_interval=2000, save_interval=40000):
        total_imgs, start_img_step = self.sess.run([self.total_imgs, self.img_step])
        start_time = dt.datetime.now()
        save_step = (total_imgs // save_interval + 1) * save_interval
        summary_step = (total_imgs // summary_interval + 1) * summary_interval
        imgs_per_stage = self.trainer.imgs_per_stage

        while True:
            self.run_train_ops()
            total_imgs, img_step = self.sess.run([self.total_imgs, self.img_step])

            if total_imgs > summary_step:
                summary_step += summary_interval
                self.run_summary_ops(start_time, start_img_step, img_step)

            if total_imgs > save_step:
                save_step += save_interval
                self.run_save_ops()

            if img_step > imgs_per_stage:
                if self.trainer.res == self.trainer.max_res and not self.trainer.is_fading:
                    return

                self.update_network()
                start_time = dt.datetime.now()
                start_img_step = 0
                imgs_per_stage = self.trainer.imgs_per_stage