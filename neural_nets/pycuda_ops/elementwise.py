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
from .. import sampler
from .matrix import extract_columns, insert_columns
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel, get_elwise_kernel

sign_kernel_float = ElementwiseKernel(
    "float *mat, float *target",
    "target[i] = (mat[i] > 0.) - (mat[i] < 0);",
    "sign")

sign_kernel_double = ElementwiseKernel(
    "double *mat, double *target",
    "target[i] = (mat[i] > 0.) - (mat[i] < 0);",
    "sign")

def sign(x):
    assert x.flags.c_contiguous
    target = gpuarray.GPUArray(x.shape, dtype=x.dtype)
    if x.dtype == np.dtype(np.float32):
        sign_kernel_float(x, target)
    elif x.dtype == np.dtype(np.float64):
        sign_kernel_double(x, target)
    else:
        raise ValueError("Incompatible dtype")
    return target

sigmoid_kernel_float = ElementwiseKernel(
        "float *mat",
        "mat[i] = 1. / (1. + __expf(-mat[i]))",
        "sigmoid")

sigmoid_kernel_double = ElementwiseKernel(
        "double *mat",
        "mat[i] = 1. / (1. + __expf(-mat[i]))",
        "sigmoid")

def sigmoid(x):
    assert x.flags.c_contiguous
    if x.dtype == np.dtype(np.float32):
        sigmoid_kernel_float(x)
    elif x.dtype == np.dtype(np.float64):
        sigmoid_kernel_double(x)
    else:
        raise ValueError("Incompatible dtype")

def df_sigmoid(f):
    assert f.flags.c_contiguous
    df = f * (1 - f)
    return df

tanh_kernel_float = ElementwiseKernel(
    "float *mat",
    "mat[i] = tanhf(mat[i]);",
    "tanh_inplace")

tanh_kernel_double = ElementwiseKernel(
    "double *mat",
    "mat[i] = tanhf(mat[i]);",
    "tanh_inplace")

def tanh(x):
    assert x.flags.c_contiguous
    if x.dtype == np.dtype(np.float32):
        tanh_kernel_float(x)
    elif x.dtype == np.dtype(np.float64):
        tanh_kernel_double(x)
    else:
        raise ValueError("Incompatible dtype")

def df_tanh(f):
    assert f.flags.c_contiguous
    df = 1 - f**2.
    return df

relu_kernel_float = ElementwiseKernel(
    "float *mat",
    "if (mat[i] < 0.) mat[i] = 0.",
    "relu")

relu_kernel_double = ElementwiseKernel(
    "double *mat",
    "if (mat[i] < 0.) mat[i] = 0.",
    "relu")

def relu(x):
    assert x.flags.c_contiguous
    if x.dtype == np.dtype(np.float32):
        relu_kernel_float(x)
    elif x.dtype == np.dtype(np.float64):
        relu_kernel_double(x)
    else:
        raise ValueError("Incompatible dtype")

df_relu_kernel_float = ElementwiseKernel(
    "float *mat, float *target",
    """if (mat[i] <= 0.) 
         target[i] = 0.;
       else
         target[i] = 1.;
    """,
    "df_relu")

df_relu_kernel_double = ElementwiseKernel(
    "double *mat, double *target",
    """if (mat[i] <= 0.) 
         target[i] = 0.;
       else
         target[i] = 1.;
    """,
    "df_relu")

def df_relu(x):
    assert x.flags.c_contiguous
    df = gpuarray.empty_like(x)
    if x.dtype == np.dtype(np.float32):
        df_relu_kernel_float(x, df)
    elif x.dtype == np.dtype(np.float64):
        df_relu_kernel_double(x, df)
    else:
        raise ValueError("Incompatible dtype")
    return df

def linear(x):
    pass

def df_linear(x):
    return x

sample_dropout_mask_kernel = get_elwise_kernel(
    "float *mat, float *dropout, float dropout_probability",
    """
    if (dropout[i] <= dropout_probability) {
         dropout[i] = 0.;
         mat[i] = 0.;
    } else {
         dropout[i] = 1.;
    }
    """,
    "sample_dropout_mask")

def sample_dropout_mask(x, dropout_probability=.5, columns=None, stream=None):
    """ Samples a dropout mask and applies it in place"""

    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])

    shape = x.shape
    dropout_mask = sampler.gen_uniform(shape, x.dtype, stream)

    sample_dropout_mask_kernel.prepared_async_call(
        dropout_mask._grid, dropout_mask._block, stream,
        x.gpudata, dropout_mask.gpudata, np.float32(dropout_probability),
        dropout_mask.size)

    if columns is not None:
        insert_columns(x, x_tmp, columns[0])
    
    return dropout_mask

apply_dropout_mask_kernel = get_elwise_kernel(
    "float *mat, float *mask",
    "if (mask[i] == 0.) mat[i] = 0;",
    "apply_dropout_mask")

def apply_dropout_mask(x, mask, columns=None, stream=None):
    assert x.flags.c_contiguous

    if columns is not None:
        assert len(columns) == 2
        x_tmp = x
        x = extract_columns(x, columns[0], columns[1])
    
    assert x.shape == mask.shape
    shape = x.shape

    apply_dropout_mask_kernel.prepared_async_call(x._grid, x._block, stream,
        x.gpudata, mask.gpudata, x.size)

    if columns is not None:
        insert_columns(x, x_tmp, columns[0])

nan_to_zeros_kernel = ElementwiseKernel("float *mat, float *target",
    "target[i] = isnan(mat[i]) ? 0. : mat[i];",
    "nan_to_zeros_kernel")

def nan_to_zeros(x, target=None):
    assert x.flags.c_contiguous
    if target is None:
        target = gpuarray.empty_like(x)
    assert target.flags.c_contiguous
    nan_to_zeros_kernel(x, target)
    return target
