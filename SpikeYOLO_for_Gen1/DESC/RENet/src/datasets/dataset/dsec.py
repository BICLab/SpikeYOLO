from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .base_dataset import BaseDataset


class DSEC(BaseDataset):
    num_classes = 1

    def __init__(self, opt, mode):
        self.ROOT_DATASET_PATH = os.path.join(opt.root_dir, 'data/DSEC')
        pkl_filename = 'DSEC-GT.pkl'
        super(DSEC, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'RGB-images', v, '{:0>6}.png'.format(i))

    def eventfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'Event-images', v, '{:0>6}.png'.format(i))

    def eventfile_30ms(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'Event-images_30ms', v, '{:0>6}.png'.format(i))
    def eventfile_50ms(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'Event-images_50ms', v, '{:0>6}.png'.format(i))
