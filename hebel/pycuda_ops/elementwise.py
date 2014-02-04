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

all_kernels = None
exp_func = None
log_func = None
def init():
    from pycuda import elementwise
    
    global all_kernels
    global exp_func
    global log_func

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

        'df_sigmoid': {
            'float': ("float *mat, float *target",
                      """const float f = mat[i];
                      target[i] = f * (1 - f);
                      """),
            'double': ("double *mat, double *target",
                       """const double f = mat[i];
                       target[i] = f * (1 - f);
                       """)
        },

        'tanh_inplace': {
            'float':  ("float *mat",
                       "mat[i] = tanhf(mat[i]);"),
            'double': ("double *mat",
                       "mat[i] = tanh(mat[i]);")
        },

        'df_tanh': {
            'float': ("float *mat, float *target",
                      """float f = mat[i];
                      target[i] = 1 - pow(f, 2);"""),
            'double': ("double *mat, double *target",
                       """double f = mat[i];
                       target[i] = 1 - pow(f, 2);""")
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
            'float':  ("float *mat, float *target, char *dropout_mask, "
                       "float *dropout_prob_array, float dropout_probability",
                       """if (dropout_prob_array[i] <= dropout_probability) {
                            dropout_mask[i] = 0.;
                            target[i] = 0.;
                          } else {
                            dropout_mask[i] = 1.;
                            if (target != mat)
                                target[i] = mat[i];
                          }
                        """),
            'double':  ("double *mat, double *targets, char *dropout_mask, "
                        "double *dropout_prob_array float dropout_probability",
                        """if (dropout_prob_array[i] <= dropout_probability) {
                            dropout_mask[i] = 0.;
                            target[i] = 0.;
                          } else {
                            dropout_mask[i] = 1.;
                            if (target != mat)                    
                                target[i] = mat[i];
                          }
                        """)
        },

        'apply_dropout_mask': {
            'float':    ("float *mat, char *mask",
                         "if (mask[i] == 0.) mat[i] = 0;"),
            'double':   ("double *mat, char *mask",
                         "if (mask[i] == 0.) mat[i] = 0;"),
        },

        'nan_to_zeros': {
            'float':    ("float *mat, float *target",
                         "target[i] = isnan(mat[i]) ? 0. : mat[i];"),
            'double':   ("double *mat, double *target",
                         "target[i] = isnan(mat[i]) ? 0. : mat[i];")
        },

        'mult_matrix': {
            'float': ("const float *a, const float *b, float *c",
                      "c[i] = a[i] * b[i];"),
            'double': ("const double *b, const double *b, double *c",
                       "c[i] = a[i] * b[i];")

        },
        'substract_matrix': {
            'float': ("const float *a, const float *b, float *c",
                      "c[i] = a[i] - b[i];"),
            'double': ("const double *a, const double *b, double *c",
                       "c[i] = a[i] - b[i];")
        }
    }

    all_kernels = {
        name: Kernel(name, 
                     val['float'][0], val['float'][1],
                     val['double'][0], val['double'][1])
        for name, val in all_kernels_code.iteritems()
    }

    exp_func = elementwise.get_unary_func_kernel('expf', np.float32)
    log_func = elementwise.get_unary_func_kernel('logf', np.float32)

def sign(x, target=None):
    assert x.flags.c_contiguous
    if target is None:
        target = gpuarray.GPUArray(x.shape, dtype=x.dtype)
    assert target.shape == x.shape
    assert target.dtype == x.dtype
    assert target.flags.c_contiguous
    all_kernels['sign'](x, target)
    return target

def sigmoid(x):
    assert x.flags.c_contiguous
    all_kernels['sigmoid'](x)

def df_sigmoid(f, target=None):
    assert f.flags.c_contiguous
    if target is None:
        target = gpuarray.empty_like(f)
    all_kernels['df_sigmoid'](f, target)
    return target

def tanh(x):
    assert x.flags.c_contiguous
    all_kernels['tanh_inplace'](x)

def df_tanh(f, target=None):
    assert f.flags.c_contiguous
    if target is None:
        target = gpuarray.empty_like(f)
    all_kernels['df_tanh'](f, target)
    return target

def relu(x):
    assert x.flags.c_contiguous
    all_kernels['relu'](x)

def df_relu(x, target=None):
    assert x.flags.c_contiguous
    if target is None:
        target = gpuarray.empty_like(x)        
    all_kernels['df_relu'](x, target)
    return target

def linear(x):
    pass

def df_linear(x):
    return x

def sample_dropout_mask(x, dropout_probability=.5, columns=None, stream=None, target=None,
                        dropout_mask=None, dropout_prob_array=None):
    """ Samples a dropout mask and applies it in place"""

    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])

    shape = x.shape

    if dropout_prob_array is None:
        dropout_prob_array = gpuarray.empty(shape, x.dtype)
    sampler.fill_uniform(dropout_prob_array, stream)

    if dropout_mask is None:
        dropout_mask = gpuarray.empty(shape, np.int8)

    if target is None: target = x
    
    all_kernels['sample_dropout_mask'](
        x, target, dropout_mask, dropout_prob_array,
        np.float32(dropout_probability))

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

def mult_matrix(a, b, target=None):
    assert a.shape == b.shape
    if target is None:
        target = gpuarray.empty_like(a)

    all_kernels['mult_matrix'](a, b, target)
    return target

def substract_matrix(a, b, target=None):
    assert a.shape == b.shape
    if target is None:
        target = gpuarray.empty_like(a)

    all_kernels['substract_matrix'](a, b, target)
    return target
