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

from sequence_convolution import enum, DNA_A, DNA_C, DNA_G, DNA_T, DNA_R, DNA_Y, DNA_N
import numpy as np
from pycuda import gpuarray
import random

Nucleotides = enum(
    A = DNA_A,
    C = DNA_C,
    G = DNA_G,
    T = DNA_T,
    R = DNA_R,
    Y = DNA_Y,
    N = DNA_N
)

def encode_nt(nt):
    return type.__getattribute__(Nucleotides, nt)

class SeqArray(object):
    def __init__(self, sequence):
        self.sequence = sequence

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, seq):
        self._sequence = seq
        self.enc_seq = gpuarray.to_gpu(
            np.array([[encode_nt(nt.upper())
                       for nt in line] for line in seq],
                       dtype=np.int8))

def sample_sequence(length, n):
    seq = [''.join((random.choice('ACGT') for i in range(length)))
           for j in range(n)]
    sa = SeqArray(seq)
    return sa

