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
from pycuda import driver as drv
from pycuda.compiler import SourceModule

code = """
__global__ void addRowVecToMat(float *mat,
                               float *vec,
                               float *target,
                               int32_t n,
                               int32_t m)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if ((ty < m) & (tx < n))
  {
      target[tx*m+ty] = vec[ty] + mat[tx*m+ty];
  }
}

__global__ void addColVecToMat(float *mat,
                               float *vec,
                               float *target,
                               int32_t n,
                               int32_t m)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if ((ty < m) & (tx < n))
  {
      target[tx*m+ty] = vec[tx] + mat[tx*m+ty];
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
"""

mod = SourceModule(code)
add_row_vec_kernel = mod.get_function('addRowVecToMat')
add_col_vec_kernel = mod.get_function('addColVecToMat')
vector_normalize_kernel = mod.get_function("kVectorNormalize")


def add_vec_to_mat(mat, vec, axis=None, inplace=False):
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

    block = (12, 12, 1)
    gridx = n // block[0] + 1 * (n % block[0] != 0)
    gridy = m // block[1] + 1 * (m % block[1] != 0)
    grid = (gridx, gridy, 1)

    if inplace:
        target = mat
    else:
        target = gpuarray.empty_like(mat)

    if axis == 0:
        assert vec.shape[0] == mat.shape[0]
        add_col_vec_kernel(mat, vec, target, np.uint32(n), np.uint32(m),
                           block=block, grid=grid)
    elif axis == 1:
        assert vec.shape[0] == mat.shape[1]
        add_row_vec_kernel(mat, vec, target, np.uint32(n), np.uint32(m),
                           block=block, grid=grid)
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


def extract_columns(mat, start=0, stop=None):
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    N, M = mat.shape
    m = stop - start

    assert mat.flags.c_contiguous
    assert start >= 0 and start <= M and stop >= 0 and \
        stop <= M and stop > start

    new_mat = gpuarray.empty((N, m), dtype)

    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.src_x_in_bytes = start * itemsize
    copy.set_dst_device(new_mat.gpudata)
    copy.src_pitch = M * itemsize
    copy.dst_pitch = copy.width_in_bytes = m * itemsize
    copy.height = N
    copy(aligned=True)

    return new_mat


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
