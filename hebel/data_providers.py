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

""" 
All data consumed by Hebel models must be provided in the form of
``DataProvider`` objects. ``DataProviders`` are classes that provide
iterators which return batches for training. By writing custom
``DataProviders```, this creates a lot of flexibility about where data
can come from and enables any sort of pre-processing on the data. For
example, a user could write a ``DataProvider`` that receives data from
the internet or through a pipe from a different process. Or, when
working with text data, a user may define a custom ``DataProvider`` to
perform tokenization and stemming on the text before returning it.

A ``DataProvider`` is defined by subclassing the
:class:`hebel.data_provider.DataProvider` class and must implement at
a minimum the special methods ``__iter__`` and ``next``.
"""

import numpy as np
from pycuda import gpuarray

class DataProvider(object):
    """ This is the abstract base class for ``DataProvider``
    objects. Subclass this class to implement a custom design. At a
    minimum you must provide implementations of the ``next`` method.
    """
    
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets

        self.N = data.shape[0]

        self.i = 0
        self.batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self._make_batches()

    def _make_batches(self):
        self.data_batches = tuple(
            self.data[i:i+self.batch_size]
            for i in range(0, self.N, self.batch_size)
        )
        self.targets_batches = tuple(
            self.targets[i:i+self.batch_size]
            for i in range(0, self.N, self.batch_size)
        )
        self.n_batches = len(self.data_batches)

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
    """ This is the standard ``DataProvider`` for mini-batch learning
    with stochastic gradient descent.

    Input and target data may either be provided as ``numpy.array``
    objects, or as ``pycuda.GPUArray`` objects. The latter is
    preferred if the data can fit on GPU memory and will be much
    faster, as the data won't have to be transferred to the GPU for
    every minibatch. If the data is provided as a ``numpy.array``,
    then every minibatch is automatically converted to to a
    ``pycuda.GPUArray`` and transferred to the GPU.

    :param data: Input data.
    :param targets: Target data.
    :param batch_size: The size of mini-batches.
    """
    
    def __getitem__(self, batch_idx):
        # return self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
        return self.batches[batch_idx]

    def next(self):
        if self.i >= self.n_batches:
            self.i = 0
            raise StopIteration

        minibatch_data  = self.data_batches[self.i]
        minibatch_targets = self.targets_batches[self.i]

        self.i += 1

        if not isinstance(minibatch_data, gpuarray.GPUArray):
            minibatch_data = gpuarray.to_gpu(minibatch_data)

        if not isinstance(minibatch_targets, gpuarray.GPUArray):
            minibatch_targets = gpuarray.to_gpu(minibatch_targets)

        return minibatch_data, minibatch_targets


class MultiTaskDataProvider(DataProvider):
    """ ``DataProvider`` for multi-task learning that uses the same
    training data for multiple targets.

    This ``DataProvider`` is similar to the
    :class:`hebel.data_provider.MiniBatchDataProvider`, except that it
    has not one but multiple targets.

    :param data: Input data.
    :param targets: Multiple targets as a list or tuple.
    :param batch_size: The size of mini-batches.

    **See also:**

    :class:`hebel.models.MultitaskNeuralNet`, :class:`hebel.layers.MultitaskTopLayer`
    
    """
    
    def __init__(self, data, targets, batch_size=None):
        if isinstance(targets, (list, tuple)):
            assert all([targets[0].shape[0] == t.shape[0] for t in targets])
        if isinstance(data, (list, tuple)):
            assert all([type(targets[0]) == type(t) for t in targets])
        self.data = data

        if not isinstance(targets, gpuarray.GPUArray):
            targets = gpuarray.to_gpu(targets)
        self.targets = targets

        try:
            self.N = data.shape[0]
        except AttributeError:
            self.N = data[0].shape[0]

        self.i = 0
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.N

    def _make_batches(self):
        if not isinstance(self.data, (list, tuple)):
            self.data_batches = tuple(
                self.data[i:i+self.batch_size]
                for i in range(0, self.N, self.batch_size)
            )
        else:
            self.data_batches = \
                tuple(tuple(d[i:i+self.batch_size]
                            for d in self.data)
                      for i in range(0, self.N, self.batch_size))

        if not isinstance(self.targets, (list, tuple)):
            self.target_batches = tuple(
                self.targets[i:i+self.batch_size]
                for i in range(0, self.N, self.batch_size)
            )
        else:
            self.target_batches = \
                tuple(tuple(tuple(t[i:i+self.batch_size]
                                  for t in self.targets))
                      for i in range(0, self.N, self.batch_size)
            )
        self.n_batches = len(self.data_batches)

    def __getitem__(self, batch_idx):
        return self.data_batches[batch_idx], self.target_batches[batch_idx]

    def next(self):
        if self.i >= self.n_batches:
            self.i = 0
            raise StopIteration

        minibatch_data = self.data_batches[self.i]
        minibatch_targets = self.target_batches[self.i]

        self.i += 1
        return minibatch_data, minibatch_targets


class BatchDataProvider(MiniBatchDataProvider):
    """``DataProvider`` for batch learning. Always returns the full data set.

    :param data: Input data.
    :param targets: Target data.

    **See also:**

    :class:`hebel.data_providers.MiniBatchDataProvider`

    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.N = data.shape[0]
        self.i = 0
        self.batch_size = self.N

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
        return self.data, self.targets

class DummyDataProvider(DataProvider):
    """A dummy ``DataProvider`` that does not store any data and
    always returns ``None``.
    """
    
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, batch_idx):
        return None, None

    def next(self):
        return None, None

class MNISTDataProvider(MiniBatchDataProvider):
    """``DataProvider`` that automatically provides data from the
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ data set of
    hand-written digits.

    Depends on the `skdata <http://jaberg.github.io/skdata/>`_ package.

    :param array: {'train', 'val', 'test'}
        Whether to use the official training, validation, or test data split of MNIST.
    :param batch_size: The size of mini-batches.
    """

    try:
        from skdata.mnist.view import OfficialVectorClassification
    except ImportError:
        from skdata.mnist.views import OfficialVectorClassification
    mnist = OfficialVectorClassification()

    def __init__(self, array, batch_size=None):

        self.train_idx = self.mnist.fit_idxs
        self.val_idx = self.mnist.val_idxs
        self.test_idx = self.mnist.tst_idxs

        self.N_train = self.train_idx.shape[0]
        self.N_val = self.val_idx.shape[0]
        self.N_test = self.test_idx.shape[0]
        self.D = self.mnist.all_vectors.shape[1]

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
            raise ValueError('Unknown partition "%s"' % array)

        self.batch_size = batch_size if batch_size is not None else self.N
        self.i = 0
        self._make_batches()

    # def __getitem__(self, batch_idx):
    #     if self.batch_size is None:
    #         if batch_idx == 0:
    #             return self.data, self.targets
    #         else:
    #             raise ValueError("batch_idx out of bounds")
    #     else:
    #         minibatch_data = self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
    #         minibatch_targets = self.targets[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
    #         return minibatch_data, minibatch_targets

    # def next(self):
    #     if self.i >= self.N:
    #         self.i = 0
    #         raise StopIteration

    #     if self.batch_size is None:
    #         self.i += self.N
    #         return self.data, self.targets
    #     else:
    #         minibatch_data = self.data[self.i:self.i+self.batch_size]
    #         minibatch_targets = self.targets[self.i:self.i+self.batch_size]
    #         self.i += self.batch_size
    #         return minibatch_data, minibatch_targets
