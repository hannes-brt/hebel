import numpy as np
from pycuda import gpuarray

""" Some basic data providers
"""

class DataProvider(object):
    def __init__(self, data, batch_size):
        self.data = data        
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

class MiniBatchDataProvider(DataProvider):
    def __getitem__(self, batch_idx):
        return self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        minibatch = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        return minibatch

# class MaxiBatchDataProvider(MiniBatchDataProvider):
#     def __init__(self, data, minibatch_size, maxibatch_size):
#         if maxibatch_size % minibatch_size:
#             raise ValueError("`maxibatch_size` must be a multiple of `minibatch_size`")
#         super(MaxiBatchDataProvider, self).__init__(data, minibatch_size)
#         self.maxibatch_size = maxibatch_size

#     def __iter__(self):
#         self.i = 0
#         self.j = 0
#         return self

#     def next(self):
#         if self.i >= self.N:
#             self.i = 0
#             raise StopIteration

#         if not i % maxibatch_size:
#             self.maxibatch = gpu.garray(self.data[self.i:self.i+self.maxibatch_size])
#             gpu.free_reuse_cache()
#             self.j = 0

#         minibatch = self.maxibatch[self.j:self.j+self.batch_size]
#         self.i += self.batch_size
#         self.j += self.batch_size

#         return minibatch

class MultiTaskDataProvider(DataProvider):
    def __init__(self, data, batch_size=None):
        assert all([data[0].shape[0] == d.shape[0] for d in data])
        assert all([type(data[0]) == type(d) for d in data])
        self.data = data

        self.N_outer = data[0].shape[0]
        self.N = self.N_outer
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.N

        # if isinstance(data[0], np.ma.MaskedArray):
        #     self.N = self.N_outer * len(self.data) - \
        #       sum([d.mask.sum() for d in self.data])
        # elif isinstance(data[0], np.ndarray) or \
        #   isinstance(data[0], gpuarray.GPUArray):
        #     self.N = self.N_outer * len(self.data) - \
        #       sum([d[0].isnan().sum() for d in self.data])

        self.i = 0

    def __getitem__(self, batch_idx):
        return [d[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                for d in self.data]

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        minibatch = [d[self.i:self.i+self.batch_size]
                     for d in self.data]
        self.i += self.batch_size
        return minibatch

# class GPUDataProvider(MiniBatchDataProvider):
#     def __getitem__(self, batch_idx):
#         minibatch = gpu.garray(super(GPUDataProvider, self).__getitem__(batch_idx))
#         gpu.free_reuse_cache()
#         return minibatch

#     def next(self):
#         minibatch = gpu.garray(super(GPUDataProvider, self).next())
#         gpu.free_reuse_cache()
#         return minibatch

class BatchDataProvider(MiniBatchDataProvider):
    def __init__(self, data):
        self.data = data        
        self.N = data.shape[0]
        self.i = 0

    def __getitem__(self, batch_idx):
        if batch_idx == 0:
            return self.data
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
        return None

    def next(self):
        return None

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
        if array == 'train_images':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.train_idx]
                                   .astype(np.float32) / 255.)
            self.N = self.N_train
        elif array == 'val_images':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.val_idx]
                                   .astype(np.float32) / 255.)
            self.N = self.N_val
        elif array == 'test_images':
            self.data = gpuarray.to_gpu(self.mnist.all_vectors[self.test_idx]
                                   .astype(np.float32) / 255.)
            self.N = self.N_test
        elif array == 'train_labels':
            targets = self.mnist.all_labels[self.train_idx]
            labels_soft = np.zeros((self.N_train, 10), dtype=np.float32)
            labels_soft[range(self.N_train), targets] = 1.
            self.data = gpuarray.to_gpu(labels_soft)
            self.N = self.N_train
        elif array == 'val_labels':
            targets = self.mnist.all_labels[self.val_idx]
            labels_soft = np.zeros((self.N_train, 10), dtype=np.float32)
            labels_soft[range(self.N_train), targets] = 1.
            self.data = gpuarray.to_gpu(labels_soft)
            self.N = self.N_val
        elif array == 'test_labels':
            targets = self.mnist.all_labels[self.test_idx]
            labels_soft = np.zeros((self.N_test, 10), dtype=np.float32)
            labels_soft[range(self.N_test), targets] = 1.
            self.data = gpuarray.to_gpu(labels_soft)
            self.N = self.N_test
        else:
            raise ValueError

        self.batch_size = batch_size if batch_size is not None else self.N
        self.i = 0

    def __getitem__(self, batch_idx):
        if self.batch_size is None:
            if batch_idx == 0:
                return self.data
            else:
                raise ValueError("batch_idx out of bounds")
        else:
            return self.data[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

    def next(self):
        if self.i >= self.N:
            self.i = 0
            raise StopIteration

        if self.batch_size is None:
            self.i += self.N
            return self.data
        else:
            minibatch = self.data[self.i:self.i+self.batch_size]
            self.i += self.batch_size
            return minibatch
