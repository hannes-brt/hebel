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
from pycuda import driver as drv
from pycuda import gpuarray
from hebel.utils.math import ceil_div

add_row_vec_kernel = None
add_col_vec_kernel = None
vector_normalize_kernel = None
_compilation_constants = {
    'add_vec_block_size': 16
}
def init():
    from pycuda.compiler import SourceModule
    
    global add_row_vec_kernel
    global add_col_vec_kernel
    global vector_normalize_kernel

    code = """
    #include <stdint.h>
    __global__ void addRowVecToMat(const float *mat,
                                   const float *vec,
                                   float *target,
                                   const int32_t n,
                                   const int32_t m,
                                   const int substract)
    {
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
      const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

      __shared__ float shared_vec[%(add_vec_block_size)d];

      if ((tx == 0) & (tidy < m))
          shared_vec[ty] = vec[tidy];
      __syncthreads();

      if ((tidy < m) & (tidx < n))
      {
          if (substract)
              target[tidx*m+tidy] = mat[tidx*m+tidy] - shared_vec[ty];
          else
              target[tidx*m+tidy] = mat[tidx*m+tidy] + shared_vec[ty];      
      }
    }

    __global__ void addColVecToMat(const float *mat,
                                   const float *vec,
                                   float *target,
                                   const int32_t n,
                                   const int32_t m,
                                   const int substract)
    {
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
      const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

      __shared__ float shared_vec[%(add_vec_block_size)d];

      if ((ty == 0) & (tidx < n))
          shared_vec[tx] = vec[tidx];
      __syncthreads();

      if ((tidy < m) & (tidx < n))
      {
          if (substract)
              target[tidx*m+tidy] = mat[tidx*m+tidy] - shared_vec[tx];
          else
              target[tidx*m+tidy] = mat[tidx*m+tidy] + shared_vec[tx];      
      }
    }

    __global__ void kVectorNormalize(float* mat,
                                     float max_vec_norm,
                                     unsigned int width,
                                     unsigned int height) {

        __shared__ float sum_shared[32];
        __shared__ float vec_norm;
        float sum = 0;

        for (unsigned int i = threadIdx.x; i < height; i += 32)
            sum += powf(mat[blockIdx.x + i * width], 2);

        sum_shared[threadIdx.x] = sum;

        __syncthreads();

        if (threadIdx.x == 0) {
            sum = 0;

            for (unsigned int i = 0; i < 32; i++)
                sum += sum_shared[i];

            vec_norm = sqrtf(sum);
        }
        __syncthreads();

        for (unsigned int i = threadIdx.x; i < height; i += 32) {
            if (vec_norm > max_vec_norm)
                mat[blockIdx.x + i * width] /= (vec_norm / max_vec_norm);
        }
    }
    """ % _compilation_constants

    mod = SourceModule(code)
    add_row_vec_kernel = mod.get_function('addRowVecToMat')
    add_col_vec_kernel = mod.get_function('addColVecToMat')
    vector_normalize_kernel = mod.get_function("kVectorNormalize")

def add_vec_to_mat(mat, vec, axis=None, inplace=False,
                   target=None, substract=False):
    """ Add a vector to a matrix
    """

    assert mat.flags.c_contiguous

    if axis is None:
        if vec.shape[0] == mat.shape[0]:
            axis = 0
        elif vec.shape[0] == mat.shape[1]:
            axis = 1
        else:
            raise ValueError('Vector length must be equal '
                             'to one side of the matrix')

    n, m = mat.shape

    block = (_compilation_constants['add_vec_block_size'],
             _compilation_constants['add_vec_block_size'], 1)
    gridx = ceil_div(n, block[0])
    gridy = ceil_div(m, block[1])
    grid = (gridx, gridy, 1)

    if inplace:
        target = mat
    elif target is None:
            target = gpuarray.empty_like(mat)

    if axis == 0:
        assert vec.shape[0] == mat.shape[0]
        add_col_vec_kernel(mat, vec, target, np.uint32(n), np.uint32(m),
                           np.int32(substract), block=block, grid=grid)
    elif axis == 1:
        assert vec.shape[0] == mat.shape[1]
        add_row_vec_kernel(mat, vec, target, np.uint32(n), np.uint32(m),
                           np.int32(substract), block=block, grid=grid)
    return target


def vector_normalize(mat, max_vec_norm=1.):
    """ Normalize each column vector in mat to length
    max_vec_norm if it is longer than max_vec_norm
    """
    assert mat.flags.c_contiguous
    n, m = mat.shape

    vector_normalize_kernel(mat, np.float32(max_vec_norm),
                            np.int32(m), np.int32(n),
                            block=(32,1,1), grid=(m,1,1))


def extract_columns(mat, start=0, stop=None, target=None):
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    N, M = mat.shape
    if stop is None:
        stop = M
    m = stop - start

    assert mat.flags.c_contiguous
    assert start >= 0 and start <= M and stop >= 0 and \
        stop <= M and stop > start

    if target is None:
        target = gpuarray.empty((N, m), dtype)

    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.src_x_in_bytes = start * itemsize
    copy.set_dst_device(target.gpudata)
    copy.src_pitch = M * itemsize
    copy.dst_pitch = copy.width_in_bytes = m * itemsize
    copy.height = N
    copy(aligned=True)

    return target


def insert_columns(src, dst, offset):
    dtype = src.dtype
    itemsize = np.dtype(dtype).itemsize
    h_src, w_src = src.shape
    h_dst, w_dst = dst.shape

    assert dst.dtype == dtype
    assert h_src == h_dst
    assert w_dst >= offset + w_src

    copy = drv.Memcpy2D()
    copy.set_src_device(src.gpudata)
    copy.set_dst_device(dst.gpudata)
    copy.dst_x_in_bytes = offset * itemsize
    copy.src_pitch = copy.width_in_bytes = w_src * itemsize
    copy.dst_pitch = w_dst * itemsize
    copy.height = h_src
    copy(aligned=True)
