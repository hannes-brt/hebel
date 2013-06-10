import numpy as np
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel, get_elwise_kernel
from pycuda.curandom import md5_code
from . import _kernel_idx

sign_kernel = [ElementwiseKernel(
    "%(dtype)s *mat, %(dtype)s *target" % {'dtype': dtype},
    "target[i] = (mat[i] > 0.) - (mat[i] < 0);",
    "sign") for dtype in ('float', 'double')]

def sign(x):
    target = gpuarray.GPUArray(x.shape, dtype=x.dtype)
    sign_kernel[_kernel_idx(x.dtype)](x, target)
    return target

sigmoid_kernel = [ElementwiseKernel(
        "%(dtype)s *mat" % {'dtype': dtype},
        "mat[i] = 1. / (1. + __expf(-mat[i]))",
        "sigmoid") for dtype in ('float', 'double')]

def sigmoid(x):
    sigmoid_kernel[_kernel_idx(x.dtype)](x)

def df_sigmoid(f):
    df = f * (1 - f)
    return df

tanh_kernel = [ElementwiseKernel(
    "%(dtype)s *mat" % {'dtype': dtype},
    "mat[i] = tanhf(mat[i]);",
    "tanh_inplace") for dtype in ('float', 'double')]

def tanh(x):
    tanh_kernel[_kernel_idx(x.dtype)](x)

def df_tanh(f):
    df = 1 - f**2.
    return df

relu_kernel = [ElementwiseKernel(
    "%(dtype)s *mat" % {'dtype': dtype},
    "if (mat[i] < 0.) mat[i] = 0.",
    "relu") for dtype in ('float', 'double')]

def relu(x):
    relu_kernel[_kernel_idx(x.dtype)](x)

df_relu_kernel = [ElementwiseKernel(
    "%(dtype)s *mat, %(dtype)s *target" % {'dtype': dtype},
    """if (mat[i] <= 0.) 
         target[i] = 0.;
       else
         target[i] = 1.;
    """,
    "df_relu") for dtype in ('float', 'double')]

def df_relu(x):
    df = gpuarray.empty_like(x)
    df_relu_kernel[_kernel_idx(x.dtype)](x, df)
    return df

sample_dropout_mask_kernel = [get_elwise_kernel(
    "%(dtype)s *mat, %(dtype)s *dest, unsigned int seed" % {'dtype': dtype},
    md5_code + """
    #define POW_2_M32 (1/4294967296.0f)

    unsigned int j = i;
    dest[j] = a*POW_2_M32;
    if ((j += total_threads) < n)
        dest[j] = b*POW_2_M32;
    if ((j += total_threads) < n)
        dest[j] = c*POW_2_M32;
    if ((j += total_threads) < n)
        dest[j] = d*POW_2_M32;

    if (dest[i] <= .5) {
         dest[i] = 0.;
         mat[i] = 0.;
    } else {
         dest[i] = 1.;
    }
    """,
    "sample_dropout_mask") for dtype in ('float', 'double')]

def sample_dropout_mask(x, stream=None):
    """ Samples a dropout mask and applies it in place"""
    
    shape = x.shape
    result = gpuarray.GPUArray(shape, np.float32)

    sample_dropout_mask_kernel[_kernel_idx(x.dtype)]\
      .prepared_async_call(
          result._grid, result._block, stream,
          x. gpudata, result.gpudata, np.random.randint(2**31-1), result.size)
    return result

apply_dropout_mask_kernel = [get_elwise_kernel(
    "%(dtype)s *mat, %(dtype)s *mask" % {'dtype': dtype},
    "if (mask[i] == 0.) mat[i] = 0;",
    "apply_dropout_mask") for dtype in ('float', 'double')]

def apply_dropout_mask(x, mask, stream=None):
    assert x.shape == mask.shape
    shape = x.shape

    apply_dropout_mask_kernel[_kernel_idx(x.dtype)]\
      .prepared_async_call(x._grid, x._block, stream,
                           x.gpudata, mask.gpudata, x.size)

nan_to_zeros_kernel = [ElementwiseKernel(
    "%(dtype)s *mat, %(dtype)s *target" % {'dtype': dtype},
    "target[i] = isnan(mat[i]) ? 0. : mat[i];",
    "nan_to_zeros_kernel") for dtype in ('float', 'double')]

def nan_to_zeros(x, target):
    nan_to_zeros_kernel[_kernel_idx(x.dtype)](x, target)
