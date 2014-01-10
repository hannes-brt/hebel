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
from hebel.data_providers import MultiTaskDataProvider


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
            self.data = [gpuarray.to_gpu(x) for x in self.data]

            if not isinstance(self.targets, (list, tuple)):
                self.targets = gpuarray.to_gpu(self.targets)
            else:
                self.targets = [gpuarray.to_gpu(t) for t in self.targets]

        if not val and self._gpu:
            self.data = [x.get() for x in self.data]

            if not isinstance(self.targets, (list, tuple)):
                self.targets = self.targets.get()
            else:
                self.targets = [t.get() for t in self.targets]        

    @property
    def sequences(self):
        return self._sequences

    @sequences.setter
    def sequences(self, seq):
        self._sequences = seq

        if not isinstance(seq[0], (list, tuple)):
            enc_seq = encode_sequence(seq)
            if self.gpu:
                enc_seq = gpuarray.to_gpu(enc_seq)
        else:
            enc_seq = [encode_sequence(s) for s in seq]
            if self.gpu:
                enc_seq = [gpuarray.to_gpu(x) for x in enc_seq]

        self.enc_seq = enc_seq


def sample_sequence(length, n):
    seq = [''.join((random.choice('ACGT') for _ in range(length)))
           for _ in range(n)]
    return seq
