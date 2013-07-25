import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scikits.cuda import linalg

code = """
#include "float.h"

__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
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

__global__ void kMaxRowwise(float* mat, float* target, unsigned int width, unsigned int height) {
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
    assert axis in (0,1)
    
    n,m = mat.shape
    
    if axis == 0:
        target = gpuarray.empty(m, dtype=np.float32)
        max_column(mat, target, np.int32(m), np.int32(n), block=(32,1,1), grid=(m,1,1))
        
    elif axis == 1:
        target = gpuarray.empty(n, dtype=np.float32)
        max_row(mat, target, np.int32(m), np.int32(n), block=(32,1,1), grid=(n,1,1))
        
    return target

def _matrix_sum_out_axis_wrapper():
    one_vector_cache = {}
    def f(mat, axis=0, cache_one_vector=True):
        N, M = mat.shape

        if axis == 0:
            vec_shape = (N,)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty(vec_shape, dtype=mat.dtype).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones
            target = linalg.dot(ones, mat).ravel()
        elif axis == 1:
            vec_shape = (M, 1)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty((M, 1), dtype=mat.dtype).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones
            target = linalg.dot(mat, ones).ravel()
        else:
            raise ValueError('axis must be 0 or 1')
        
        return target
    return f
matrix_sum_out_axis = _matrix_sum_out_axis_wrapper()

