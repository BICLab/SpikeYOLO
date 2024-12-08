from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.sampler import Sampler

from .dataset.dsec import DSEC


switch_dataset = {
    'dsec': DSEC,
}


def get_dataset(dataset):
    class Dataset(switch_dataset[dataset], Sampler):
        pass
    return Dataset
