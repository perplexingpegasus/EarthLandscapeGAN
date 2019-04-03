from data.array_maker import ArrayMaker
from data.restorable import Restorable, ArrayHandler

from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np


def normal(size):
    return np.random.normal(0.0, 1.0, size=size)

def uniform(size):
    return np.random.uniform(-1.0, 1.0, size=size)

def spherical(size):
    uniform = np.random.uniform(-1.0, 1.0, size=size)
    norm = np.linalg.norm(uniform, axis=0, keepdims=True)
    return uniform / norm

def truncated_normal(size):
    return truncnorm.rvs(-2.0, 2.0, size=size)


class ScheduledParam:

    def __init__(self, stable_params, fading_params=None):
        self.stable_params = stable_params

        if fading_params is None:
            self.fading_params = stable_params

        else:
            assert all([i in stable_params.keys() for i in fading_params.keys()])
            self.fading_params = fading_params


class ModelTrainer(Restorable, ArrayHandler):
    filename = 'model_trainer.pkl'

    def __init__(self, logdir, arraydir, channels, imgs_per_stage, batch_size, z_length=None, z_distribution='normal',
        z_fixed_size=30, random_state=0, **model_params):

        if self.is_saved(logdir):
            return

        am = ArrayMaker.load(arraydir)
        assert len(channels) >= len(am.sizes)
        Restorable.__init__(self, logdir)
        ArrayHandler.__init__(self, arraydir, am.sizes, am.max_imgs, am.total_n_imgs)

        self.channels = channels
        self.model_params = dict(imgs_per_stage=imgs_per_stage, batch_size=batch_size, **model_params)

        if z_length is None:
            self.z_length = channels[0]
        else:
            self.z_length = z_length

        if z_distribution == 'normal':
            self.z_rvf = normal
        elif z_distribution == 'uniform':
            self.z_rvf = uniform
        elif z_distribution == 'spherical':
            self.z_rvf = spherical
        elif z_distribution == 'truncated':
            self.z_rvf = truncated_normal
        else:
            raise ValueError("z_rvf must be one of (normal, uniform, spherical, truncated)")

        np.random.seed(random_state)
        self.z_fixed = self.z_batch(z_fixed_size)

        self.res = self.min_res
        self.is_fading = False
        self.master_idx = 0

        self.array = self.load_array(self.res, 0)

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super().__getattr__(name)

        try:
            item = self.model_params[name]
        except KeyError:
            raise AttributeError(name)

        if isinstance(item, ScheduledParam):
            if self.is_fading:
                return item.fading_params[self.res]
            else:
                return item.stable_params[self.res]
        else:
            return item

    @property
    def n_blocks(self):
        return int(np.log2(self.res / self.min_res)) + 1

    def z_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return self.z_rvf([batch_size, self.z_length])

    def x_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        start_master_idx = self.master_idx
        _, start_img_idx = self.get_idx(self.res, start_master_idx)
        batch = self.array[start_img_idx:start_img_idx + batch_size]

        cur_batch_size = batch.shape[0]
        self.master_idx = (start_master_idx + cur_batch_size) % self.total_n_imgs
        remaining = batch_size - cur_batch_size

        while remaining > 0:
            self._load_array()
            batch = np.concatenate((batch, self.array[:remaining]))
            cur_batch_size = batch.shape[0]
            self.master_idx = (start_master_idx + cur_batch_size) % self.total_n_imgs
            remaining = batch_size - cur_batch_size

        return batch

    def change_stage(self, res=None, is_fading=None):
        if res is None and is_fading is None:
            if not self.is_fading:
                if self.res == self.max_res:
                    return

                self.res *= 2

            self.is_fading = not self.is_fading

        if res is not None:
            assert res in self.sizes
            self.res = res

        if is_fading is not None:
            self.is_fading = is_fading

        self._load_array()

    def _load_array(self):
        array_idx, _ = self.get_idx(self.res, self.master_idx)
        self.array = self.load_array(self.res, array_idx)

    def save(self):
        del self.array
        super().save()
        self._load_array()

    @classmethod
    def load(cls, logdir):
        restored_mt = super().load(logdir)
        restored_mt._load_array()
        return restored_mt