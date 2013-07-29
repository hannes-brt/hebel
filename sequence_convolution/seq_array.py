from sequence_convolution import enum
import numpy as np
from pycuda import gpuarray

Nucleotides = enum(
    A = 0b1000,
    C = 0b0100,
    G = 0b0010,
    T = 0b0001,
    R = 0b1010,
    Y = 0b0101,
    N = 0b0000
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
            np.array([[encode_nt(nt)
                       for nt in line] for line in seq],
                       dtype=np.int8))

