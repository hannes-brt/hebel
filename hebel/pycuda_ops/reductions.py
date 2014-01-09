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
from . import linalg

max_column = None
max_row = None
def init():
    from pycuda.compiler import SourceModule

    global max_column
    global max_row

    code = """
#include "float.h"

__global__ void kMaxColumnwise(float* mat,
                               float* target,
                               unsigned int width,
                               unsigned int height) {
    __shared__ float max_vals[32];
    float cur_max = -FLT_MAX;
    float val = 0;

    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x + i * width];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
    // __syncthreads();
}

__global__ void kMaxRowwise(float* mat,
                            float* target,
                            unsigned int width,
                            unsigned int height) {
    __shared__ float max_vals[32];
    float cur_max = -FLT_MAX;
    float val = 0;

    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[blockIdx.x * width + i];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
    // __syncthreads();
}
"""

    mod = SourceModule(code)
    max_column = mod.get_function("kMaxColumnwise")
    max_row = mod.get_function("kMaxRowwise")


def max_by_axis(mat, axis=0):
    assert mat.flags.c_contiguous
    assert axis in (0, 1)

    n, m = mat.shape

    if axis == 0:
        target = gpuarray.empty(m, dtype=np.float32)
        max_column(mat, target, np.int32(m), np.int32(n),
                   block=(32, 1, 1), grid=(m, 1, 1))

    elif axis == 1:
        target = gpuarray.empty(n, dtype=np.float32)
        max_row(mat, target, np.int32(m), np.int32(n),
                block=(32, 1, 1), grid=(n, 1, 1))

    return target


def _matrix_sum_out_axis_wrapper():
    one_vector_cache = {}

    def f(mat, axis=0, cache_one_vector=True, target=None):
        assert mat.flags.c_contiguous
        N, M = mat.shape

        if axis == 0:
            vec_shape = (N, 1)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty(vec_shape, dtype=mat.dtype).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones

            if target is None:
                target = gpuarray.empty((M,), mat.dtype)

            # if len(target.shape) == 1:
                # target = target.reshape((target.shape[0], 1))
                # target.shape = (target.shape[0], 1)
            assert target.shape == (M,)
            linalg.dot(mat, ones, transa='T', target=target)
        elif axis == 1:
            vec_shape = (M, 1)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty((M, 1), dtype=mat.dtype).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones

            if target is None:
                target = gpuarray.empty((N,), mat.dtype)

            # if len(target.shape) == 1:
            #     target = target.reshape((target.shape[0], 1))
            assert target.shape == (N,)
            linalg.dot(mat, ones, target=target)
        else:
            raise ValueError('axis must be 0 or 1')

        # target.shape = (target.shape[0], 1)
        return target
    return f
matrix_sum_out_axis = _matrix_sum_out_axis_wrapper()
