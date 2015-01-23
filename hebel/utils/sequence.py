import numpy as np
import string
import random

def encode_sequence(seq):
    seq_upper = map(string.upper, seq)
    enc_seq = np.array(seq_upper, 'c')
    return enc_seq

def sample_sequence(length, n):
    seq = [''.join((random.choice('ACGT') for _ in range(length)))
           for _ in range(n)]
    return seq
