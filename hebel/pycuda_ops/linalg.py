# This file is modified from scikits.cuda (https://github.com/lebedov/scikits.cuda)
# Copyright (c) 2009-2013, Lev Givon. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# Neither the name of Lev Givon nor the names of any contributors may
# be used to endorse or promote products derived from this software
# without specific prior written permission.  THIS SOFTWARE IS
# PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

from string import lower
import pycuda.gpuarray as gpuarray
import numpy as np
from . import cublas

def init():
    global _global_cublas_handle
    _global_cublas_handle = cublas.cublasCreate()

def dot(x_gpu, y_gpu, transa='N', transb='N', handle=None, target=None):
    """
    Dot product of two arrays.

    For 1D arrays, this function computes the inner product. For 2D
    arrays of shapes `(m, k)` and `(k, n)`, it computes the matrix
    product; the result has shape `(m, n)`.

    Parameters
    ----------
    x_gpu : pycuda.gpuarray.GPUArray
        Input array.
    y_gpu : pycuda.gpuarray.GPUArray
        Input array.
    transa : char
        If 'T', compute the product of the transpose of `x_gpu`.
        If 'C', compute the product of the Hermitian of `x_gpu`.
    transb : char
        If 'T', compute the product of the transpose of `y_gpu`.
        If 'C', compute the product of the Hermitian of `y_gpu`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `scikits.cuda.misc._global_cublas_handle` is used.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray, float{32,64}, or complex{64,128}
        Inner product of `x_gpu` and `y_gpu`. When the inputs are 1D
        arrays, the result will be returned as a scalar.

    Notes
    -----
    The input matrices must all contain elements of the same data type.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import linalg
    >>> import misc
    >>> linalg.init()
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> b = np.asarray(np.random.rand(2, 2), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> c_gpu = linalg.dot(a_gpu, b_gpu)
    >>> np.allclose(np.dot(a, b), c_gpu.get())
    True
    >>> d = np.asarray(np.random.rand(5), np.float32)
    >>> e = np.asarray(np.random.rand(5), np.float32)
    >>> d_gpu = gpuarray.to_gpu(d)
    >>> e_gpu = gpuarray.to_gpu(e)
    >>> f = linalg.dot(d_gpu, e_gpu)
    >>> np.allclose(np.dot(d, e), f)
    True

    """

    if handle is None:
        handle = _global_cublas_handle

    if len(x_gpu.shape) == 1 and len(y_gpu.shape) == 1:

        if x_gpu.size != y_gpu.size:
            raise ValueError('arrays must be of same length')

        # Compute inner product for 1D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas.cublasCdotu
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas.cublasSdot
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas.cublasZdotu
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas.cublasDdot
        else:
            raise ValueError('unsupported combination of input types')

        return cublas_func(handle, x_gpu.size, x_gpu.gpudata, 1,
                           y_gpu.gpudata, 1)
    else:

        # Get the shapes of the arguments (accounting for the
        # possibility that one of them may only have one dimension):
        x_shape = x_gpu.shape
        y_shape = y_gpu.shape
        if len(x_shape) == 1:
            x_shape = (1, x_shape[0])
        if len(y_shape) == 1:
            y_shape = (1, y_shape[0])

        # Perform matrix multiplication for 2D arrays:
        if (x_gpu.dtype == np.complex64 and y_gpu.dtype == np.complex64):
            cublas_func = cublas.cublasCgemm
            alpha = np.complex64(1.0)
            beta = np.complex64(0.0)
        elif (x_gpu.dtype == np.float32 and y_gpu.dtype == np.float32):
            cublas_func = cublas.cublasSgemm
            alpha = np.float32(1.0)
            beta = np.float32(0.0)
        elif (x_gpu.dtype == np.complex128 and y_gpu.dtype == np.complex128):
            cublas_func = cublas.cublasZgemm
            alpha = np.complex128(1.0)
            beta = np.complex128(0.0)
        elif (x_gpu.dtype == np.float64 and y_gpu.dtype == np.float64):
            cublas_func = cublas.cublasDgemm
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
        else:
            raise ValueError('unsupported combination of input types')

        transa = lower(transa)
        transb = lower(transb)

        if transb in ['t', 'c']:
            m, k = y_shape
        elif transb in ['n']:
            k, m = y_shape
        else:
            raise ValueError('invalid value for transb')

        if transa in ['t', 'c']:
            l, n = x_shape
        elif transa in ['n']:
            n, l = x_shape
        else:
            raise ValueError('invalid value for transa')

        if l != k:
            raise ValueError('objects are not aligned')

        if transb == 'n':
            lda = max(1, m)
        else:
            lda = max(1, k)

        if transa == 'n':
            ldb = max(1, k)
        else:
            ldb = max(1, n)

        ldc = max(1, m)

        # Note that the desired shape of the output matrix is the transpose
        # of what CUBLAS assumes:

        if target is None:
            target = gpuarray.empty((n, ldc), x_gpu.dtype)
        
        cublas_func(handle, transb, transa, m, n, k, alpha, y_gpu.gpudata,
                    lda, x_gpu.gpudata, ldb, beta, target.gpudata, ldc)

        return target
