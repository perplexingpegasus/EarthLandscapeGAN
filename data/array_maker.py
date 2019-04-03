from data.restorable import ArrayHandler, Restorable

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


class ArrayMaker(Restorable, ArrayHandler):
    filename = 'array_maker.pkl'

    def __init__(self, arraydir, imgdir=None, min_res=4, max_res=1024, max_file_size=1.0, crop_offset=0.25,
        use_mirror=True, use_blur=True, NCHW=True):

        if self.is_saved(arraydir):
            return

        else:
            if imgdir is None:
                raise ValueError('"imgdir" must be provided if model is not being restored')

        res_exps = []
        for res in (min_res, max_res):
            res_exp = np.log2(res)
            if res_exp % 1 != 0:
                raise ValueError('resolutions must be powers of 2')
            res_exps.append(int(res_exp))
        min_res_exp, max_res_exp = res_exps

        sizes = [2 ** i for i in range(min_res_exp, max_res_exp + 1)]
        max_imgs = {s: int(max_file_size * 1e9) // (3 * s ** 2) for s in sizes}

        Restorable.__init__(self, arraydir)
        ArrayHandler.__init__(self, arraydir, sizes, max_imgs, 0)

        self.imgdir = imgdir
        self.use_mirror = use_mirror
        self.use_blur = use_blur
        self.NCHW = NCHW
        self.swap_channels = True
        self.processed_img_files = []

        if crop_offset is not None:
            self.hop_length = int(self.max_res * crop_offset)
            if self.hop_length <= 0 or self.hop_length > self.max_res:
                raise ValueError("crop offset must be > 0 and <= 1 or 'None'")
        else:
            self.hop_length = None

    def load_array(self, res, array_idx, suffix=''):
        array = super().load_array(res, array_idx, suffix)
        if self.NCHW and self.swap_channels:
            array = np.transpose(array, (0, 2, 3, 1))
        return array

    def save_array(self, array, array_idx, suffix=''):
        res = array.shape[2]
        if self.NCHW and self.swap_channels:
            array = np.transpose(array, (0, 3, 1, 2))
        np.save(self.get_array_filename(res, array_idx, suffix), array)

    def remove_array(self, res, array_idx, suffix=''):
        os.remove(self.get_array_filename(res, array_idx, suffix))

    def get_new_array(self, res):
        max_imgs = self.max_imgs[res]
        if self.NCHW and not self.swap_channels:
            shape = [max_imgs, 3, res, res]
        else:
            shape = [max_imgs, res, res, 3]
        return np.zeros(shape, dtype=np.uint8)

    def initialize_loop_conditions(self, res):
        self.cur_array = self.get_new_array(res)
        self.new_n_imgs = 0

        if self.total_n_imgs == 0:
            self.array_idx, self.img_idx = 0, 0

        else:
            self.array_idx, self.img_idx = self.get_idx(res)
            if self.img_idx != 0:
                self.cur_array[:self.img_idx] = self.load_array(res, self.array_idx)

    def insert_img(self, img, res, crop_window=None):
        if crop_window is not None:
            img = img.crop(crop_window)
            img = np.asarray(img, dtype=np.uint8)
        self.cur_array[self.img_idx] = img

        self.new_n_imgs += 1
        self.array_idx, self.img_idx = self.get_idx(res, self.total_n_imgs + self.new_n_imgs)

        if self.img_idx == 0:
            self.save_array(self.cur_array, self.array_idx - 1)

    def end_loop_save(self):
        if self.img_idx != 0:
            self.save_array(self.cur_array[:self.img_idx], self.array_idx)

    def make_arrays(self):
        self.initialize_loop_conditions(self.max_res)
        hr_start_array_idx, hr_start_img_idx = self.array_idx, self.img_idx

        first_run = len(self.processed_img_files) == 0
        img_files = os.listdir(self.imgdir)
        if not first_run: img_files = list(set(self.processed_img_files) - set(img_files))
        n_imgfs = len(img_files)

        for i, imgf in enumerate(img_files):
            print('Processing file {}/{} {}\n\n'.format(i, n_imgfs, imgf))

            try:
                with Image.open(os.path.join(self.imgdir, imgf)) as img:
                    w, h = img.size
                    if w < self.max_res or h < self.max_res:
                        print('Cannot use file {} because it has dimensions '
                              'less than maximum resolution\n\n'.format(imgf))
                        continue

                    img = img.convert('RGB')
                    landscape = w > h

                    if landscape:
                        w = int(w / h * self.max_res)
                        h = self.max_res
                    else:
                        h = int(h / w * self.max_res)
                        w = self.max_res

                    img = img.resize((w, h), Image.LANCZOS)

            except Exception:
                print('Error when processing file {}\n\n'.format(imgf))

            else:
                if landscape:
                    get_window = lambda x: (x, 0, h + x, h)
                else:
                    get_window = lambda x: (0, x, w, w + x)

                base_imgs = [img]
                if self.use_mirror:
                    mirror_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    base_imgs.append(mirror_img)

                for base_img in base_imgs:

                    if self.hop_length is not None:
                        n_crops = abs(h - w) // self.hop_length + 1

                        for i in range(n_crops):
                            window = get_window(i * self.hop_length)
                            self.insert_img(base_img, self.max_res, window)
                    else:
                        window = get_window(abs(h - w) // 2)
                        self.insert_img(base_img, self.max_res, window)

        self.end_loop_save()
        hr_end_array_idx = self.array_idx + (0 if self.img_idx == 0 else 1)

        if self.new_n_imgs == 0:
            print('No new image files could be processed\n\n')
            return

        new_total_n_imgs = self.total_n_imgs + self.new_n_imgs
        rand_seq = list(range(new_total_n_imgs))
        np.random.shuffle(rand_seq)

        if first_run:
            self.shuffle_arrays(self.max_res, rand_seq, new_total_n_imgs)

        img_placeholder = tf.placeholder(tf.uint8)
        downpool_op = self.downpool(img_placeholder)
        sess = tf.Session()

        self.swap_channels = False

        higher_res = self.max_res
        for res in reversed(self.sizes[:-1]):
            print('resizing {}x{} images\n\n'.format(res, res))
            self.initialize_loop_conditions(res)
            start_array_idx, start_img_idx = self.array_idx, self.img_idx

            for i in range(hr_start_array_idx, hr_end_array_idx):
                print(i)
                transfer_array = self.load_array(higher_res, i)
                start = hr_start_img_idx if i == hr_start_array_idx else 0

                lr_array = sess.run(downpool_op, {img_placeholder: transfer_array[start:]})
                for img in lr_array:
                    self.insert_img(img, res)

            self.end_loop_save()
            del self.cur_array
            higher_res = res
            hr_start_array_idx, hr_start_img_idx = start_array_idx, start_img_idx
            hr_end_array_idx = self.array_idx + (0 if self.img_idx == 0 else 1)

        self.swap_channels = True
        self.processed_img_files += img_files
        self.total_n_imgs = new_total_n_imgs
        self.save()

        if not first_run:
            for res in reversed(self.sizes):
                if res != self.max_res:
                    self.shuffle_arrays(res, rand_seq)

        self.save()

    def depthwise_conv(self, imgs, f, k=1):
        f = np.reshape(f, [*f.shape, 1, 1])
        f = np.tile(f, [1, 1, 3, 1])
        f = tf.constant(f, name='filter', dtype=tf.float32)

        if self.NCHW:
            data_format = 'NCHW'
            strides = [1, 1, k, k]
        else:
            data_format = 'NHWC'
            strides = [1, k, k, 1]

        return tf.nn.depthwise_conv2d(imgs, f, strides, padding='SAME', data_format=data_format)

    def downpool(self, imgs):
        imgs = tf.cast(imgs, tf.float32)

        if self.use_blur:
            f = np.array([
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]
            ], dtype=np.float32)
            with tf.variable_scope('blur'):
                imgs = self.depthwise_conv(imgs, f)

        f = np.array([
            [0.25, 0.25],
            [0.25, 0.25]
        ], dtype=np.float32)
        with tf.variable_scope('downpool'):
            imgs = self.depthwise_conv(imgs, f, 2)

        return tf.cast(imgs, tf.uint8)

    def shuffle_arrays(self, res, rand_seq, new_total_n_imgs=None):
        print('shuffling {}x{} arrays\n\n'.format(res, res))
        self.array_idx, self.img_idx = self.get_idx(res, new_total_n_imgs)

        if self.array_idx == 0:
            unshuffled_array = self.load_array(res, 0)
            shuffled_array = unshuffled_array[rand_seq]
            del unshuffled_array
            self.save_array(shuffled_array, 0)
            del shuffled_array

        else:
            max_imgs = self.max_imgs[res]
            n_arrays = self.array_idx + (1 if self.img_idx != 0 else 0)
            sub_seqs = [rand_seq[i * max_imgs : (i + 1) * max_imgs] for i in range(n_arrays)]
            recomb_seqs = [[] for _ in range(n_arrays)]
            array_idx_seqs = [[] for _ in range(n_arrays)]

            for i in range(n_arrays):
                unshuffled_array = self.load_array(res, i)
                start = i * max_imgs
                end = start + max_imgs

                for j in range(n_arrays):
                    selection = [k for k in sub_seqs[j] if start <= k and k < end]
                    recomb_seqs[j] += selection
                    selection = [k % max_imgs for k in selection]

                    if len(selection) != 0:
                        array_idx_seqs[j].append(i)
                        sub_array = unshuffled_array[selection]
                        self.save_array(sub_array, j, '_{:05}'.format(i))
                        del sub_array

                del unshuffled_array
                self.remove_array(res, i)

            for i in range(n_arrays):
                sub_arrays = []

                for j in array_idx_seqs[i]:
                    sub_arrays.append(self.load_array(res, i, '_{:05}'.format(j)))
                    self.remove_array(res, i, '_{:05}'.format(j))

                new_array = np.concatenate(sub_arrays)
                del sub_arrays

                selection = [recomb_seqs[i].index(j) for j in sub_seqs[i]]
                new_array = new_array[selection]
                self.save_array(new_array, i)
                del new_array

    def preview_images(self, start, stop):
        n_imgs = stop - start
        for i, res in enumerate(self.sizes):
            plt.figure(i)
            max_imgs = self.max_imgs[res]
            arr_idx, img_idx = divmod(start, max_imgs)
            cur_array = np.load(os.path.join(self.logdir, '{:05d}_{:05d}.npy'.format(res, arr_idx)))
            big_img = np.zeros([res, n_imgs * res, 3], np.uint8)
            for j in range(n_imgs):
                big_img[:, j * res : (j + 1) * res] = np.transpose(cur_array[img_idx], (1, 2, 0))
                img_idx += 1
                if img_idx == max_imgs:
                    arr_idx += 1
                    img_idx = 0
                    cur_array = np.load(os.path.join(self.logdir, '{:05d}_{:05d}.npy'.format(res, arr_idx)))
            plt.imshow(big_img)
            plt.show()