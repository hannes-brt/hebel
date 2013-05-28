import numpy as np
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel, get_elwise_kernel
from pycuda.curandom import md5_code

sign_kernel = ElementwiseKernel(
    "float *mat, float *target",
    "target[i] = (mat[i] > 0.) - (mat[i] < 0);",
    "sign")

def sign(x):
    target = gpuarray.GPUArray(x.shape, dtype=x.dtype)
    sign_kernel(x, target)
    return target

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

sample_dropout_mask_kernel = get_elwise_kernel(
    "float *mat, float *dest, unsigned int seed",
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
    "sample_dropout_mask")

def sample_dropout_mask(x, stream=None):
    """ Samples a dropout mask and applies it in place"""
    
    shape = x.shape
    result = gpuarray.GPUArray(shape, np.float32)

    sample_dropout_mask_kernel.prepared_async_call(
        result._grid, result._block, stream,
        x. gpudata, result.gpudata, np.random.randint(2**31-1), result.size)
    return result

apply_dropout_mask_kernel = get_elwise_kernel(
    "float *mat, float *mask",
    "if (mask[i] == 0.) mat[i] = 0;",
    "apply_dropout_mask")

def apply_dropout_mask(x, mask, stream=None):
    assert x.shape == mask.shape
    shape = x.shape

    apply_dropout_mask_kernel.prepared_async_call(x._grid, x._block, stream,
        x.gpudata, mask.gpudata, x.size)

nan_to_zeros_kernel = ElementwiseKernel("float *mat, float *target",
    "target[i] = isnan(mat[i]) ? 0. : mat[i];",
    "nan_to_zeros_kernel")
