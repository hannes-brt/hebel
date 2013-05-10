import numpy as np
from pycuda import gpuarray, cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

from scikits.cuda import linalg

eps = np.finfo(np.float32).eps

sigmoid_kernel = ElementwiseKernel(
        "float *mat",
        "mat[i] = 1. / (1. + __expf(-mat[i]))",
        "sigmoid")

def df_sigmoid(f):
    df = f * (1 - f)
    return df

tanh_kernel = ElementwiseKernel(
    "float *mat",
    "mat[i] = tanhf(mat[i]);",
    "tanh_inplace")

def df_tanh(f):
    df = 1 - f**2.
    return df

relu_kernel = ElementwiseKernel(
    "float *mat",
    "if (mat[i] < 0.) mat[i] = 0.",
    "relu")

df_relu_kernel = ElementwiseKernel(
    "float *mat, float *target",
    """if (mat[i] <= 0.) 
         target[i] = 0.;
       else
         target[i] = 1.;
    """,
    "df_relu")

def df_relu(x):
    df = gpuarray.empty_like(x)
    df_relu_kernel(x, df)
    return df

code = """
#include "float.h"

__global__ void addRowVecToMat(float *mat, float *vec, float *target, int32_t n, int32_t m)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
   
  if ((ty < m) & (tx < n))
  {
      target[tx*m+ty] = vec[ty] + mat[tx*m+ty];
  }
}

__global__ void addColVecToMat(float *mat, float *vec, float *target, int32_t n, int32_t m)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
   
  if ((ty < m) & (tx < n))
  {
      target[tx*m+ty] = vec[tx] + mat[tx*m+ty];
  }
}

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
    __syncthreads();
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
    __syncthreads();
}

"""
mod = SourceModule(code)
add_row_vec_kernel = mod.get_function('addRowVecToMat')
add_col_vec_kernel = mod.get_function('addColVecToMat')
max_column = mod.get_function("kMaxColumnwise")
max_row = mod.get_function("kMaxRowwise")

def add_vec_to_mat(mat, vec, axis=None, inplace=False):
    """ Add a vector to a matrix
    """
    
    if axis is None:
        if vec.shape[0] == mat.shape[0]: 
            axis = 0
        elif vec.shape[0] == mat.shape[1]:
            axis = 1
        else:
            raise ValueError('Vector length must be equal to one side of the matrix')            
    
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

def logsumexp(mat):
    max_dim = max_by_axis(mat, 1)
    tmp = add_vec_to_mat(mat, -max_dim, 0)
    L = max_dim + cumath.log(matrix_sum_out_axis(cumath.exp(tmp), 1))
    return L

def softmax(mat):
    L = logsumexp(mat)
    return cumath.exp(add_vec_to_mat(mat, -L, inplace=True))

def cross_entropy(x, y):
    loss = y * cumath.log(x + eps)
    loss = -gpuarray.sum(loss)
    return float(loss.get())

def _matrix_sum_out_axis_wrapper():
    one_vector_cache = {}
    def f(mat, axis=0, cache_one_vector=True):
        N, M = mat.shape

        if axis == 0:
            vec_shape = (N,)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty(vec_shape, dtype=np.float32).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones
            target = linalg.dot(ones, mat).ravel()
        elif axis == 1:
            vec_shape = (M, 1)
            try:
                ones = one_vector_cache[vec_shape]
            except KeyError:
                ones = gpuarray.empty((M, 1), dtype=np.float32).fill(1.)
                if cache_one_vector: one_vector_cache[vec_shape] = ones
            target = linalg.dot(mat, ones).ravel()
        else:
            raise ValueError('axis must be 0 or 1')
        
        return target
    return f
matrix_sum_out_axis = _matrix_sum_out_axis_wrapper()
