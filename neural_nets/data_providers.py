# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

""" Some basic data providers
"""

import numpy as np
from pycuda import gpuarray

class DataProvider(object):
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size

        self.N = data.shape[0]

        self.i = 0

    def __getitem__(self, batch_idx):
        raise NotImplementedError

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.data.shape


class MiniBatchDataProvider(DataProvider):
    def __getitem__(self, batch_idx):
        return self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        minibatch_data  = self.data[self.i:self.i+self.batch_size]
        minibatch_targets = self.targets[self.i:self.i+self.batch_size]

        self.i += self.batch_size

        if not isinstance(minibatch_data, gpuarray.GPUArray):
            minibatch_data = gpuarray.to_gpu(minibatch_data)

        if not isinstance(minibatch_targets, gpuarray.GPUArray):
            minibatch_targets = gpuarray.to_gpu(minibatch_targets)

        return minibatch_data, minibatch_targets


class MultiTaskDataProvider(DataProvider):
    def __init__(self, data, targets, batch_size=None):
        assert all([targets[0].shape[0] == t.shape[0] for t in targets])
        assert all([type(targets[0]) == type(t) for t in targets])
        self.data = data
        self.targets = targets

        try:
            self.N = data.shape[0]
        except AttributeError:
            self.N = data[0].shape[0]

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.N

        self.i = 0

    def __getitem__(self, batch_idx):
        if not isinstance(self.data, (list, tuple)):
            minibatch_data = \
              self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            if not isinstance(minibatch_data, gpuarray.GPUArray):
                minibatch_data = gpuarray.to_gpu(minibatch_data)
        else:
            minibatch_data = \
              [d[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
               for d in self.data]
            if not isinstance(minibatch_data[0], gpuarray.GPUArray):
                minibatch_data = [gpuarray.to_gpu(d) for d in minibatch_data]
        if not isinstance(self.data, (list, tuple)):
            minibatch_targets = \
              self.targets[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            if not isinstance(minibatch_targets, gpuarray.GPUArray):
                minibatch_targets = gpuarray.to_gpu(minibatch_targets)
        else:
            minibatch_targets = \
              [t[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
               for t in self.targets]
            if not isinstance(minibatch_targets[0], gpuarray.GPUArray):
                minibatch_targets = [gpuarray.to_gpu(d) for d in minibatch_targets]
        return minibatch_data, minibatch_targets

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        if not isinstance(self.data, (list, tuple)):
            minibatch_data = self.data[self.i:self.i+self.batch_size]
            if not isinstance(minibatch_data, gpuarray.GPUArray):
                minibatch_data = gpuarray.to_gpu(minibatch_data)
        else:
            minibatch_data = \
              [d[self.i:self.i+self.batch_size]
               for d in self.data]
            if not isinstance(minibatch_data[0], gpuarray.GPUArray):
                minibatch_data = [gpuarray.to_gpu(d) for d in minibatch_data]

        if not isinstance(self.targets, (list, tuple)):
            minibatch_targets = self.targets[self.i:self.i+self.batch_size]
            if not isinstance(minibatch_targets, gpuarray.GPUArray):
                minibatch_targets = gpuarray.to_gpu(minibatch_targets)
        else:
            minibatch_targets = [t[self.i:self.i+self.batch_size]
                                 for t in self.targets]
            if not isinstance(minibatch_targets[0], gpuarray.GPUArray):
                minibatch_targets = [gpuarray.to_gpu(d) for d in minibatch_targets]
        self.i += self.batch_size
        return minibatch_data, minibatch_targets


class BatchDataProvider(MiniBatchDataProvider):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.N = data.shape[0]
        self.i = 0

    def __getitem__(self, batch_idx):
        if batch_idx == 0:
            return self.data, self.targets
        else:
            raise ValueError("batch_idx out of bounds")

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        self.i += self.N
        return self.data

class DummyDataProvider(DataProvider):
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, batch_idx):
        return None, None

    def next(self):
        return None, None

class MNISTDataProvider(DataProvider):
    from skdata.mnist.views import OfficialVectorClassification
    mnist = OfficialVectorClassification()

    train_idx = mnist.fit_idxs
    val_idx = mnist.val_idxs
    test_idx = mnist.tst_idxs

    N_train = train_idx.shape[0]
    N_val = val_idx.shape[0]
    N_test = test_idx.shape[0]
    D = mnist.all_vectors.shape[1]

    def __init__(self, array, batch_size=None):
        if array == 'train':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.train_idx]
                                   .astype(np.float32) / 255.)
            targets = self.mnist.all_labels[self.train_idx]
            labels_soft = np.zeros((self.N_train, 10), dtype=np.float32)
            labels_soft[range(self.N_train), targets] = 1.
            self.targets = gpuarray.to_gpu(labels_soft)
            self.N = self.N_train
        elif array == 'val':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.val_idx]
                                   .astype(np.float32) / 255.)
            self.N = self.N_val
            targets = self.mnist.all_labels[self.val_idx]
            labels_soft = np.zeros((self.N_train, 10), dtype=np.float32)
            labels_soft[range(self.N_val), targets] = 1.
            self.targets = gpuarray.to_gpu(labels_soft)
        elif array == 'test':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.test_idx]
                                   .astype(np.float32) / 255.)
            targets = self.mnist.all_labels[self.test_idx]
            labels_soft = np.zeros((self.N_test, 10), dtype=np.float32)
            labels_soft[range(self.N_test), targets] = 1.
            self.targets = gpuarray.to_gpu(labels_soft)
            self.N = self.N_test
        else:
            raise ValueError

        self.batch_size = batch_size if batch_size is not None else self.N
        self.i = 0

    def __getitem__(self, batch_idx):
        if self.batch_size is None:
            if batch_idx == 0:
                return self.data, self.targets
            else:
                raise ValueError("batch_idx out of bounds")
        else:
            minibatch_data = self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            minibatch_targets = self.targets[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            return minibatch_data, minibatch_targets

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        if self.batch_size is None:
            self.i += self.N
            return self.data, self.targets
        else:
            minibatch_data = self.data[self.i:self.i+self.batch_size]
            minibatch_targets = self.targets[self.i:self.i+self.batch_size]
            self.i += self.batch_size
            return minibatch_data, minibatch_targets
