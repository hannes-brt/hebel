import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scikits.cuda import linalg

code = """
#include "float.h"

__global__ void kMaxColumnwise(%(data_type)s* mat, %(data_type)s* target, 
    unsigned int width, unsigned int height) {

    __shared__ %(data_type)s max_vals[32];
    %(data_type)s cur_max = -FLT_MAX;
    %(data_type)s val = 0;
 
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

__global__ void kMaxRowwise(%(data_type)s* mat, %(data_type)s* target, 
    unsigned int width, unsigned int height) {
    
    __shared__ %(data_type)s max_vals[32];
    %(data_type)s cur_max = -FLT_MAX;
    %(data_type)s val = 0;
 
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

mod = [SourceModule(code % {'data_type': dtype}) for dtype in ('float', 'double')]
max_column = [m.get_function("kMaxColumnwise") for m in mod]
max_row = [m.get_function("kMaxRowwise") for m in mod]

def max_by_axis(mat, axis=0):
    assert axis in (0,1)
    dtype = mat.dtype
    assert dtype in (np.float32, np.float64)
    n,m = mat.shape

    kernel = max_column[0] if dtype == np.float32 else max_column[1]
    
    if axis == 0:
        target = gpuarray.empty(m, dtype=dtype)
        kernel(mat, target, np.int32(m), np.int32(n), block=(32,1,1), grid=(m,1,1))
        
    elif axis == 1:
        target = gpuarray.empty(n, dtype=dtype)
        kernel(mat, target, np.int32(m), np.int32(n), block=(32,1,1), grid=(n,1,1))
        
    return target

def _matrix_sum_out_axis_wrapper():
    one_vector_cache = {}
    def f(mat, axis=0, cache_one_vector=True):
        N, M = mat.shape
        dtype = mat.dtype

        if axis == 0:
            vec_shape = (N,)
            try:
                ones = one_vector_cache[dtype][vec_shape]
            except KeyError:
                ones = gpuarray.empty(vec_shape, dtype=dtype).fill(1.)
                if cache_one_vector: 
                    if not one_vector_cache.has_key(dtype):
                        one_vector_cache[dtype] = {}
                    one_vector_cache[dtype][vec_shape] = ones
            target = linalg.dot(ones, mat).ravel()
        elif axis == 1:
            vec_shape = (M, 1)
            try:
                ones = one_vector_cache[dtype][vec_shape]
            except KeyError:
                ones = gpuarray.empty((M, 1), dtype=dtype).fill(1.)
                if cache_one_vector: 
                    if not one_vector_cache.has_key(dtype):
                        one_vector_cache[dtype] = {}
                    one_vector_cache[vec_shape] = ones
            target = linalg.dot(mat, ones).ravel()
        else:
            raise ValueError('axis must be 0 or 1')
        
        return target
    return f
matrix_sum_out_axis = _matrix_sum_out_axis_wrapper()

