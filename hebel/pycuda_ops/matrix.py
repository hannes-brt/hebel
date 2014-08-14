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

from .. import memory_pool, sampler
import numpy as np
from pycuda import driver as drv
from pycuda import gpuarray
from ..utils.math import ceil_div

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
                                   const unsigned int n,
                                   const unsigned int m,
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
                                   const unsigned int n,
                                   const unsigned int m,
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
    add_row_vec_kernel = mod.get_function('addRowVecToMat').prepare('PPPIIi')
    add_col_vec_kernel = mod.get_function('addColVecToMat').prepare('PPPIIi')
    vector_normalize_kernel = mod.get_function("kVectorNormalize").prepare('PfII')

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
        add_col_vec_kernel.prepared_call(
            grid, block,
            mat.gpudata,
            vec.gpudata,
            target.gpudata,
            np.uint32(n),
            np.uint32(m),
            np.int32(substract))
    elif axis == 1:
        assert vec.shape[0] == mat.shape[1]
        add_row_vec_kernel.prepared_call(
            grid, block,
            mat.gpudata,
            vec.gpudata,
            target.gpudata,
            np.uint32(n),
            np.uint32(m),
            np.int32(substract))
    return target


def vector_normalize(mat, max_vec_norm=1.):
    """ Normalize each column vector in mat to length
    max_vec_norm if it is longer than max_vec_norm
    """
    assert mat.flags.c_contiguous
    n, m = mat.shape

    vector_normalize_kernel.prepared_call(
        (m, 1, 1), (32, 1, 1),
        mat.gpudata,
        np.float32(max_vec_norm),
        np.int32(m),
        np.int32(n))

def extract_columns(mat, start=0, stop=None, target=None):
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize

    input_3d = False
    if len(mat.shape) == 2:
        N, M = mat.shape
        if stop is None:
            stop = M
    elif len(mat.shape) == 3:
        input_3d = True
        N, M, Z = mat.shape
        if stop is None:
            stop = M
        start = start * Z
        stop = stop * Z
        M = M * Z
        mat = mat.reshape((N, M))
    else:
        raise ValueError("mat must have two or three dimensions")
    m = stop - start

    assert mat.flags.c_contiguous
    assert start >= 0 and start <= M and stop >= 0 and \
        stop <= M and stop > start

    if target is None:
        target = gpuarray.empty((N, m), dtype, allocator=memory_pool.allocate)

    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.src_x_in_bytes = start * itemsize
    copy.set_dst_device(target.gpudata)
    copy.src_pitch = M * itemsize
    copy.dst_pitch = copy.width_in_bytes = m * itemsize
    copy.height = N
    copy(aligned=True)

    if input_3d:
        assert not m % Z
        target = target.reshape((N, m // Z, Z))

    return target


def insert_columns(src, dst, offset):
    dtype = src.dtype
    itemsize = np.dtype(dtype).itemsize
    if len(src.shape) == 2:
        h_src, w_src = src.shape
    elif len(src.shape) == 3:
        h_src = src.shape[0]
        w_src = np.prod(src.shape[1:])
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

def pad_array(mat, left=0, right=0, val=0., new_shape=None, stream=None):
    assert mat.flags.c_contiguous

    is_chararray = False
    if mat.dtype == '|S1':
        is_chararray = True
        mat.dtype = np.int8
        if type(val) is str:
            val = ord(val)
    
    if len(mat.shape) == 2:
        height, width = mat.shape
    elif len(mat.shape) > 2:
        height = mat.shape[0]
        width = np.prod(mat.shape[1:])
        mat = mat.reshape((height, width))
    else:
        raise ValueError('Array must be at least two-dimensional.')

    padded_width = width + left + right

    padded_mat = gpuarray.empty((height, padded_width), dtype=mat.dtype,
                                allocator=memory_pool.allocate).fill(val)

    itemsize = np.dtype(padded_mat.dtype).itemsize
    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.set_dst_device(padded_mat.gpudata)
    copy.dst_x_in_bytes = left * itemsize
    copy.src_pitch = copy.width_in_bytes = width * itemsize
    copy.dst_pitch = padded_width * itemsize
    copy.height = height
    copy(stream)

    if new_shape is not None:
        padded_mat = padded_mat.reshape(new_shape)

    if is_chararray:
        mat.dtype = np.dtype('|S1')
        padded_mat.dtype = np.dtype('|S1')
        
    return padded_mat
    
def rand_array(shape, dtype=np.float32, dist='uniform', stream=None):
    mat = gpuarray.empty(shape, dtype, allocator=memory_pool.allocate)
    if dist == 'uniform':
        sampler.fill_uniform(mat, stream=stream)
    elif dist == 'normal':
        sampler.fill_normal(mat, stream=stream)
    return mat
