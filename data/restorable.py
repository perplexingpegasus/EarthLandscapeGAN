import os
import pickle
import numpy as np


class Restorable:
    filename = 'restorable_object.pkl'

    def __new__(cls, logdir=None, *args, **kwargs):
        if logdir is not None:
            if cls.is_saved(logdir):
                return cls.load(logdir)
        return super().__new__(cls)

    def __init__(self, logdir, *args, **kwargs):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    @classmethod
    def get_path(cls, logdir):
        return os.path.join(logdir, cls.filename)

    @classmethod
    def is_saved(cls, logdir):
        return os.path.exists(cls.get_path(logdir))

    @classmethod
    def load(cls, logdir):
        with open(cls.get_path(logdir), 'rb') as f:
            return pickle.load(f)

    def save(self):
        with open(self.get_path(self.logdir), 'wb') as f:
            pickle.dump(self, f)


class ArrayHandler:

    def __init__(self, arraydir, sizes, max_imgs, total_n_imgs, *args, **kwargs):
        self.arraydir = arraydir
        self.sizes = sizes
        self.max_imgs = max_imgs
        self.total_n_imgs = total_n_imgs
        self.min_res = min(self.sizes)
        self.max_res = max(self.sizes)

        if not os.path.exists(self.arraydir):
            os.makedirs(self.arraydir)

    def get_idx(self, res, n_imgs=None):
        if n_imgs is None:
            n_imgs = self.total_n_imgs
        return divmod(n_imgs, self.max_imgs[res])

    def get_array_filename(self, res, array_idx, suffix=''):
        return os.path.join(self.arraydir, '{:05d}_{:05d}{}.npy'.format(res, array_idx, suffix))

    def load_array(self, res, array_idx, suffix=''):
        return np.load(self.get_array_filename(res, array_idx, suffix))