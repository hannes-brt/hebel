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
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from .. import sampler
from .matrix import extract_columns, insert_columns

all_kernels_code = {
    'sign': {
        'float':  ("float *mat, float *target",
                   "target[i] = (mat[i] > 0.) - (mat[i] < 0);"),
        'double': ("double *mat, double *target",
                   "target[i] = (mat[i] > 0.) - (mat[i] < 0);")
        },

    'sigmoid': {
        'float':  ("float *mat",
                   "mat[i] = 1. / (1. + __expf(-mat[i]))",),
        'double': ("double *mat",
                   "mat[i] = 1. / (1. + exp(-mat[i]))")
        },

    'tanh': {
        'float':  ("float *mat",
                   "mat[i] = tanhf(mat[i]);"),
        'double': ("double *mat",
                   "mat[i] = tanh(mat[i]);")
        },

    'relu': {
        'float':  ("float *mat",
                   "if (mat[i] < 0.) mat[i] = 0.",),
        'double': ("double *mat",
                   "if (mat[i] < 0.) mat[i] = 0.")
        },

    'df_relu': {
        'float':  ("float *mat, float *target",
                   "if (mat[i] <= 0.)\n  target[i] = 0.;\nelse\n  target[i] = 1.;"),
        'double': ("double *mat, double *target",
                   "if (mat[i] <= 0.)\n  target[i] = 0.;\nelse\n  target[i] = 1.;")
        },

    'sample_dropout_mask': {
        'float':  ("float *mat, float *dropout, float dropout_probability",
                   """if (dropout[i] <= dropout_probability) {
                        dropout[i] = 0.;
                        mat[i] = 0.;
                      } else {
                        dropout[i] = 1.;
                      }
                    """),
        'double':  ("double *mat, double *dropout, float dropout_probability",
                    """if (dropout[i] <= dropout_probability) {
                        dropout[i] = 0.;
                        mat[i] = 0.;
                      } else {
                        dropout[i] = 1.;
                      }
                    """)
        },

    'apply_dropout_mask': {
        'float':    ("float *mat, float *mask",
                     "if (mask[i] == 0.) mat[i] = 0;"),
        'double':   ("double *mat, double *mask",
                     "if (mask[i] == 0.) mat[i] = 0;"),
        },

    'nan_to_zeros': {
        'float':    ("float *mat, float *target",
                     "target[i] = isnan(mat[i]) ? 0. : mat[i];"),
        'double':   ("double *mat, double *target",
                     "target[i] = isnan(mat[i]) ? 0. : mat[i];")
        }
}

class Kernel(object):
    """ Defers creation of the ElementwiseKernels until the first
    runtime and automatically selects kernels for double and float.
    """

    def __init__(self, name, signature_float, code_float, 
                 signature_double, code_double):
        self.name = name
        self.kernel_float = ElementwiseKernel(signature_float, code_float, name)
        self.kernel_double = ElementwiseKernel(signature_double, code_double, name)

    def __call__(self, *args, **kwargs):
        if args[0].dtype == np.float32:
            self.kernel_float(*args, **kwargs)
        elif args[0].dtype == np.float64:
            self.kernel_double(*args, **kwargs)
        else:
            raise ValueError("Unknown datatype, must be np.float32 or np.float64")

    def get_kernel(self, dtype):
        if dtype == np.float32 or dtype == 'float':
            return self.kernel_float
        elif dtype == np.float64 or dtype == 'double':
            return self.kernel_double
        else:
            raise ValueError("Unknown datatype, must be np.float32 or np.float64")

all_kernels = {
    name: Kernel(name, 
                 val['float'][0], val['float'][1],
                 val['double'][0], val['double'][1])
    for name, val in all_kernels_code.iteritems()
}
        

def sign(x):
    assert x.flags.c_contiguous
    target = gpuarray.GPUArray(x.shape, dtype=x.dtype)
    all_kernels['sign'](x, target)
    return target

def sigmoid(x):
    assert x.flags.c_contiguous
    all_kernels['sigmoid'](x)

def df_sigmoid(f):
    assert f.flags.c_contiguous
    df = f * (1 - f)
    return df

def tanh(x):
    assert x.flags.c_contiguous
    all_kernels['tanh'](x)

def df_tanh(f):
    assert f.flags.c_contiguous
    df = 1 - f ** 2.
    return df

def relu(x):
    assert x.flags.c_contiguous
    all_kernels['relu'](x)

def df_relu(x):
    assert x.flags.c_contiguous
    df = gpuarray.empty_like(x)
    all_kernels['df_relu'](x, df)
    return df

def linear(x):
    pass

def df_linear(x):
    return x

def sample_dropout_mask(x, dropout_probability=.5, columns=None, stream=None):
    """ Samples a dropout mask and applies it in place"""

    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])

    shape = x.shape
    dropout_mask = sampler.gen_uniform(shape, x.dtype, stream)

    all_kernels['sample_dropout_mask'](
        x, dropout_mask, np.float32(dropout_probability))

    if columns is not None:
        insert_columns(x, x_tmp, columns[0])

    return dropout_mask

def apply_dropout_mask(x, mask, columns=None, stream=None):
    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])

    assert x.shape == mask.shape
    shape = x.shape

    all_kernels['apply_dropout_mask'](x, mask)

    if columns is not None:
        insert_columns(x, x_tmp, columns[0])

def nan_to_zeros(x, target=None):
    assert x.flags.c_contiguous
    if target is None:
        target = gpuarray.empty_like(x)
    assert target.flags.c_contiguous
    all_kernels['nan_to_zeros'](x, target)
    return target
