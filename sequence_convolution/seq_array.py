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


import numpy as np
import string
from pycuda import gpuarray
import random
from hebel import memory_pool
from hebel.data_providers import MultiTaskDataProvider
from hebel.utils.math import ceil_div


def encode_sequence(seq):
    seq_upper = map(string.upper, seq)
    enc_seq = np.array(seq_upper, 'c')
    return enc_seq


class SeqArrayDataProvider(MultiTaskDataProvider):
    def __init__(self, sequences,
                 targets, batch_size,
                 data_inputs=None,
                 event_id=None,
                 tissue=None,
                 psi_mean=None,
                 gpu=True,
                 **kwargs):

        self.event_id = event_id
        self.tissue = tissue
        self.psi_mean = psi_mean

        self._gpu = gpu
        self.sequences = sequences

        if data_inputs is not None:
            data = self.enc_seq + data_inputs
        else:
            data = self.enc_seq

        for key, value in kwargs.iteritems():
            self.__dict__[key] = value
        super(SeqArrayDataProvider, self).__init__(data, targets, batch_size)

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, val):
        if val and not self._gpu:
            self.data = [gpuarray.to_gpu(x, allocator=memory_pool.allocate) for x in self.data]

            if not isinstance(self.targets, (list, tuple)):
                self.targets = gpuarray.to_gpu(self.targets, allocator=memory_pool.allocate)
            else:
                self.targets = [gpuarray.to_gpu(t, allocator=memory_pool.allocate) for t in self.targets]

        if not val and self._gpu:
            self.data = [x.get() for x in self.data]

            if not isinstance(self.targets, (list, tuple)):
                self.targets = self.targets.get()
            else:
                self.targets = [t.get() for t in self.targets]

        self._make_batches()

    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, seq):
        self._sequences = seq

        if not isinstance(seq[0], (list, tuple)):
            enc_seq = encode_sequence(seq)
            if self.gpu:
                enc_seq = gpuarray.to_gpu(enc_seq, allocator=memory_pool.allocate)
        else:
            enc_seq = [encode_sequence(s) for s in seq]
            if self.gpu:
                enc_seq = [gpuarray.to_gpu(x, allocator=memory_pool.allocate) for x in enc_seq]

        self.enc_seq = enc_seq


class HDF5SeqArrayDataProvider(MultiTaskDataProvider):
    def __init__(self, group, split_name, batch_size=None, trim=None):
        self.group = group
        self.table = self.group._f_get_child(split_name)
        self.batch_size = self.group._v_attrs.batch_size \
                          if batch_size is None else batch_size
        self.n_pos_batch = self.group._v_attrs.n_pos_batch
        self.n_neg_batch = self.group._v_attrs.n_neg_batch
        self.seq_length = self.table.cols.seq.dtype.itemsize - \
                          (sum(trim) if trim is not None else 0)
        self.trim = trim

        self.N = self.table.nrows
        self.i = 0

        self.get_next_batch()
        self.n_batches = ceil_div(self.N, self.batch_size)
    
    def _make_batches(self):
        pass

    def next(self):
        if self.i >= self.N:
            self.i = 0
            self.get_next_batch()
            raise StopIteration
        
        # self.table.read(self.i, self.i + self.batch_size)
        # self.sequences = gpuarray.to_gpu(
        #     np.ndarray((self.batch_size, self.seq_length),
        #                '|S1', np.ascontiguousarray(self.batch['seq']).data))
        # self.targets = gpuarray.to_gpu(np.ascontiguousarray(self.batch['label'], np.float32)
        #                                .reshape((self.batch_size, 1)))

        sequences = self.sequences_next
        targets = self.targets_next

        self.i += self.batch_size
        self.get_next_batch()

        assert sequences.shape[0] > 0
        return [sequences], targets

    def get_next_batch(self):
        if self.i < self.N:
            data = self.table.read(self.i, self.i + self.batch_size)
            assert data.shape[0] > 0

            self.sequences_next = \
                np.ndarray((data.shape[0], data['seq'].itemsize),
                           '|S1', np.ascontiguousarray(data['seq']).data)
            if self.trim:
                self.sequences_next = np.copy(self.sequences_next[:, self.trim[0]:-self.trim[1]])
            self.sequences_next = gpuarray.to_gpu_async(
                self.sequences_next, allocator=memory_pool.allocate)
            self.targets_next = gpuarray.to_gpu_async(
                np.ascontiguousarray(data['label'], np.float32)
                .reshape((data.shape[0], 1)),
                allocator=memory_pool.allocate
            )


def sample_sequence(length, n):
    seq = [''.join((random.choice('ACGT') for _ in range(length)))
           for _ in range(n)]
    return seq
