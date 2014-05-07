# This file is taken from scikits.cuda (https://github.com/lebedov/scikits.cuda)
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

#!/usr/bin/env python

"""
Python interface to CUBLAS functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import re
import sys
import warnings
import ctypes
import ctypes.util
import atexit
import numpy as np

from string import Template

import cuda
import utils


if sys.platform == 'linux2':
    _libcublas_libname_list = ['libcublas.so', 'libcublas.so.4', 'libcublas.so.5']
elif sys.platform == 'darwin':
    _libcublas_libname_list = ['libcublas.dylib']
elif sys.platform == 'win32':
    _libcublas_libname_list = ['cublas64_60.dll', 'cublas32_60.dll', 
                               'cublas64_55.dll', 'cublas32_55.dll', 
                               'cublas64_50.dll', 'cublas32_50.dll']
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcublas = None
for _libcublas_libname in _libcublas_libname_list:
    try:
        _libcublas = ctypes.cdll.LoadLibrary(_libcublas_libname)
    except OSError:
        pass
    else:
        break
if _libcublas == None:
    raise OSError('cublas library not found')

# Generic CUBLAS error:
class cublasError(Exception):
    """CUBLAS error"""
    pass

# Exceptions corresponding to different CUBLAS errors:
class cublasNotInitialized(cublasError):
    """CUBLAS library not initialized."""
    pass

class cublasAllocFailed(cublasError):
    """Resource allocation failed."""
    pass

class cublasInvalidValue(cublasError):
    """Unsupported numerical value was passed to function."""
    pass

class cublasArchMismatch(cublasError):
    """Function requires an architectural feature absent from the device."""
    pass

class cublasMappingError(cublasError):
    """Access to GPU memory space failed."""
    pass

class cublasExecutionFailed(cublasError):
    """GPU program failed to execute."""
    pass

class cublasInternalError(cublasError):
    """An internal CUBLAS operation failed."""
    pass

cublasExceptions = {
    0x1: cublasNotInitialized,
    0x3: cublasAllocFailed,
    0x7: cublasInvalidValue,
    0x8: cublasArchMismatch,
    0xb: cublasMappingError,
    0xd: cublasExecutionFailed,
    0xe: cublasInternalError,
    }

_CUBLAS_OP = {
    0: 0,   # CUBLAS_OP_N
    'n': 0, 
    'N': 0,
    1: 1,   # CUBLAS_OP_T
    't': 1, 
    'T': 1,
    2: 2,   # CUBLAS_OP_C
    'c': 2, 
    'C': 2,
    }

_CUBLAS_FILL_MODE = {
    0: 0,   # CUBLAS_FILL_MODE_LOWER
    'l': 0, 
    'L': 0,
    1: 1,   # CUBLAS_FILL_MODE_UPPER
    'u': 1, 
    'U': 1,
    }

_CUBLAS_DIAG = {
    0: 0,   # CUBLAS_DIAG_NON_UNIT,
    'n': 0, 
    'N': 0,
    1: 1,   # CUBLAS_DIAG_UNIT
    'u': 1, 
    'U': 1,
    }

_CUBLAS_SIDE_MODE = {
    0: 0,   # CUBLAS_SIDE_LEFT
    'l': 0,
    'L': 0, 
    1: 1,   # CUBLAS_SIDE_RIGHT
    'r': 1,
    'r': 1  
    }

def cublasCheckStatus(status):
    """
    Raise CUBLAS exception
    
    Raise an exception corresponding to the specified CUBLAS error
    code.
    
    Parameters
    ----------
    status : int
        CUBLAS error code.

    See Also
    --------
    cublasExceptions

    """
    
    if status != 0:
        try:
            raise cublasExceptions[status]
        except KeyError:
            raise cublasError

# Helper functions:

_libcublas.cublasCreate_v2.restype = int
_libcublas.cublasCreate_v2.argtypes = [ctypes.c_void_p]
def cublasCreate():
    """
    Initialize CUBLAS.

    Initializes CUBLAS and creates a handle to a structure holding
    the CUBLAS library context.

    Returns
    -------
    handle : int
        CUBLAS context.
            
    """

    handle = ctypes.c_int()
    status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
    cublasCheckStatus(status)
    return handle.value    

_libcublas.cublasDestroy_v2.restype = int
_libcublas.cublasDestroy_v2.argtypes = [ctypes.c_int]
def cublasDestroy(handle):
    """
    Release CUBLAS resources.

    Releases hardware resources used by CUBLAS.

    Parameters
    ----------
    handle : int
        CUBLAS context.
        
    """

    status = _libcublas.cublasDestroy_v2(ctypes.c_int(handle))
    cublasCheckStatus(status)

_libcublas.cublasGetVersion_v2.restype = int
_libcublas.cublasGetVersion_v2.argtypes = [ctypes.c_int,
                                           ctypes.c_void_p]
def cublasGetVersion(handle):
    """
    Get CUBLAS version.

    Returns version number of installed CUBLAS libraries.

    Parameters
    ----------
    handle : int
        CUBLAS context.

    Returns
    -------
    version : int
        CUBLAS version.

    """
    
    version = ctypes.c_int()
    status = _libcublas.cublasGetVersion_v2(handle, ctypes.byref(version))
    cublasCheckStatus(status)
    return version.value

_libcublas.cublasSetStream_v2.restype = int
_libcublas.cublasSetStream_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_int]
def cublasSetStream(handle, id):
    """
    Set current CUBLAS library stream.
    
    Parameters
    ----------
    handle : id
        CUBLAS context.
    id : int
        Stream ID.

    """

    status = _libcublas.cublasSetStream_v2(handle, id)
    cublasCheckStatus(status)

_libcublas.cublasGetStream_v2.restype = int
_libcublas.cublasGetStream_v2.argtypes = [ctypes.c_int,
                                          ctypes.c_void_p]
def cublasGetStream(handle):
    """
    Set current CUBLAS library stream.

    Parameters
    ----------
    handle : int
        CUBLAS context.
  
    Returns
    -------
    id : int
        Stream ID.
  
    """
    
    id = ctypes.c_int()
    status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(id))
    cublasCheckStatus(status)
    return id.value

try:
    _libcublas.cublasGetCurrentCtx.restype = int
except AttributeError:
    def cublasGetCurrentCtx():
        raise NotImplementedError(
            'cublasGetCurrentCtx() not found; CULA CUBLAS library probably\n'
            'precedes NVIDIA CUBLAS library in library search path')
else:
    def cublasGetCurrentCtx():
        """
        Get current CUBLAS context.
        
        Returns the current context used by CUBLAS.
        
        Returns
        -------
        handle : int
            CUBLAS context.
        
        """

        return _libcublas.cublasGetCurrentCtx()
    
### BLAS Level 1 Functions ###

# ISAMAX, IDAMAX, ICAMAX, IZAMAX
I_AMAX_doc = Template(
"""
    Index of maximum magnitude element.

    Finds the smallest index of the maximum magnitude element of a
    ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of maximum magnitude element.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data} 
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> m = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(m, np.argmax(np.abs(x)))
    True
    
    Notes
    -----
    This function returns a 0-based index.
    
""")

_libcublas.cublasIsamax_v2.restype = int
_libcublas.cublasIsamax_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIsamax(handle, n, x, incx):
    result = ctypes.c_int()    
    status = \
           _libcublas.cublasIsamax_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIsamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='single-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float32)',
                                           func='cublasIsamax')

_libcublas.cublasIdamax_v2.restype = int
_libcublas.cublasIdamax_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIdamax(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIdamax_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIdamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='double-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float64)',
                                           func='cublasIdamax')

_libcublas.cublasIcamax_v2.restype = int
_libcublas.cublasIcamax_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIcamax(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIcamax_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIcamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='single precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                           func='cublasIcamax')

_libcublas.cublasIzamax_v2.restype = int
_libcublas.cublasIzamax_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIzamax(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIzamax_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1
    
cublasIzamax.__doc__ = \
                     I_AMAX_doc.substitute(precision='double precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                           func='cublasIzamax')

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
I_AMIN_doc = Template(
"""
    Index of minimum magnitude element (${precision} ${real}).

    Finds the smallest index of the minimum magnitude element of a
    ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of minimum magnitude element.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> m = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(m, np.argmin(x))
    True

    Notes
    -----
    This function returns a 0-based index.

    """
)

_libcublas.cublasIsamin_v2.restype = int
_libcublas.cublasIsamin_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIsamin(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIsamin_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIsamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='single-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float32)',
                                           func='cublasIsamin')

_libcublas.cublasIdamin_v2.restype = int
_libcublas.cublasIdamin_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIdamin(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIdamin_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIdamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='double-precision',
                                           real='real',
                                           data='np.random.rand(5).astype(np.float64)',
                                           func='cublasIdamin')

_libcublas.cublasIcamin_v2.restype = int
_libcublas.cublasIcamin_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIcamin(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIcamin_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIcamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='single-precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                           func='cublasIcamin')

_libcublas.cublasIzamin_v2.restype = int
_libcublas.cublasIzamin_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasIzamin(handle, n, x, incx):
    result = ctypes.c_int()
    status = \
           _libcublas.cublasIzamin_v2(handle,
                                      n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value-1

cublasIzamin.__doc__ = \
                     I_AMIN_doc.substitute(precision='double-precision',
                                           real='complex',
                                           data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                           func='cublasIzamin')

# SASUM, DASUM, SCASUM, DZASUM
_ASUM_doc = Template(                    
"""
    Sum of absolute values of ${precision} ${real} vector.

    Computes the sum of the absolute values of the elements of a
    ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> s = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(s, np.sum(np.abs(x)))
    True

    Returns
    -------
    s : ${ret_type}
        Sum of absolute values.
        
    """
)

_libcublas.cublasSasum_v2.restype = int
_libcublas.cublasSasum_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasSasum(handle, n, x, incx):
    result = ctypes.c_float()
    status = _libcublas.cublasSasum_v2(handle,
                                       n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)

cublasSasum.__doc__ = \
                    _ASUM_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSasum',
                                         ret_type='numpy.float32')

_libcublas.cublasDasum_v2.restype = int
_libcublas.cublasDasum_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasDasum(handle, n, x, incx):
    result = ctypes.c_double()
    status = _libcublas.cublasDasum_v2(handle,
                                       n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)

cublasDasum.__doc__ = \
                    _ASUM_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDasum',
                                         ret_type='numpy.float64')

_libcublas.cublasScasum_v2.restype = int
_libcublas.cublasScasum_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasScasum(handle, n, x, incx):
    result = ctypes.c_float()
    status = _libcublas.cublasScasum_v2(handle,
                                        n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)
    
cublasScasum.__doc__ = \
                     _ASUM_doc.substitute(precision='single-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                          func='cublasScasum',
                                          ret_type='numpy.float32')

_libcublas.cublasDzasum_v2.restype = int
_libcublas.cublasDzasum_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasDzasum(handle, n, x, incx):
    result = ctypes.c_double()
    status = _libcublas.cublasDzasum_v2(handle,
                                        n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)

cublasDzasum.__doc__ = \
                     _ASUM_doc.substitute(precision='double-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          func='cublasDzasum',
                                          ret_type='numpy.float64')

# SAXPY, DAXPY, CAXPY, ZAXPY
_AXPY_doc = Template(
"""
    Vector addition (${precision} ${real}).

    Computes the sum of a ${precision} ${real} vector scaled by a
    ${precision} ${real} scalar and another ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    alpha : ${type}
        Scalar.
    x : ctypes.c_void_p
        Pointer to single-precision input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to single-precision input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = ${alpha} 
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> ${func}(h, x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
    """
)

_libcublas.cublasSaxpy_v2.restype = int
_libcublas.cublasSaxpy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSaxpy(handle, n, alpha, x, incx, y, incy):
    status = _libcublas.cublasSaxpy_v2(handle,
                                       n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasSaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='single-precision',
                                         real='real',
                                         type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSaxpy')

_libcublas.cublasDaxpy_v2.restype = int
_libcublas.cublasDaxpy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDaxpy(handle, n, alpha, x, incx, y, incy):
    status = _libcublas.cublasDaxpy_v2(handle,
                                       n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasDaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='double-precision',
                                         real='real',
                                         type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDaxpy')

_libcublas.cublasCaxpy_v2.restype = int
_libcublas.cublasCaxpy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCaxpy(handle, n, alpha, x, incx, y, incy):
    status = _libcublas.cublasCaxpy_v2(handle, n,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
                                       int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasCaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='single-precision',
                                         real='complex',
                                         type='numpy.complex64',
                                         alpha='(np.random.rand()+1j*np.random.rand()).astype(np.complex64)',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',             
                                         func='cublasCaxpy')

_libcublas.cublasZaxpy_v2.restype = int
_libcublas.cublasZaxpy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZaxpy(handle, n, alpha, x, incx, y, incy):
    status = _libcublas.cublasZaxpy_v2(handle, n,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
                                       int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasZaxpy.__doc__ = \
                    _AXPY_doc.substitute(precision='double-precision',
                                         real='complex',
                                         type='numpy.complex128',
                                         alpha='(np.random.rand()+1j*np.random.rand()).astype(np.complex128)',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',             
                                         func='cublasZaxpy')

# SCOPY, DCOPY, CCOPY, ZCOPY
_COPY_doc = Template(
"""
    Vector copy (${precision} ${real})

    Copies a ${precision} ${real} vector to another ${precision} ${real}
    vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.zeros_like(x_gpu)
    >>> h = cublasCreate()
    >>> ${func}(h, x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(y_gpu.get(), x_gpu.get())
    True
    
    Notes
    -----
    Both `x` and `y` must contain `n` elements.

""")

_libcublas.cublasScopy_v2.restype = int
_libcublas.cublasScopy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasScopy(handle, n, x, incx, y, incy):
    status = _libcublas.cublasScopy_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)
                
cublasScopy.__doc__ = \
                    _COPY_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasScopy')

_libcublas.cublasDcopy_v2.restype = int
_libcublas.cublasDcopy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDcopy(handle, n, x, incx, y, incy):
    status = _libcublas.cublasDcopy_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)
                
cublasDcopy.__doc__ = \
                    _COPY_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDcopy')

_libcublas.cublasCcopy_v2.restype = int
_libcublas.cublasCcopy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCcopy(handle, n, x, incx, y, incy):
    status = _libcublas.cublasCcopy_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)
                
cublasCcopy.__doc__ = \
                    _COPY_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+np.random.rand(5).astype(np.complex64)',
                                         func='cublasCcopy')

_libcublas.cublasZcopy_v2.restype = int
_libcublas.cublasZcopy_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZcopy(handle, n, x, incx, y, incy):
    status = _libcublas.cublasZcopy_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)
                
cublasZcopy.__doc__ = \
                    _COPY_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+np.random.rand(5).astype(np.complex128)',
                                         func='cublasZcopy')

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_DOT_doc = Template(
"""
    Vector dot product (${precision} ${real})

    Computes the dot product of two ${precision} ${real} vectors.
    cublasCdotc and cublasZdotc use the conjugate of the first vector
    when computing the dot product.
    
    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Returns
    -------
    d : ${ret_type}
        Dot product of `x` and `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> d = ${func}(x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> ${check} 
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.
    
""")

_libcublas.cublasSdot_v2.restype = int
_libcublas.cublasSdot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasSdot(handle, n, x, incx, y, incy):
    result = ctypes.c_float()
    status = _libcublas.cublasSdot_v2(handle, n,
                                      int(x), incx, int(y), incy,
                                      ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)

cublasSdot.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.float32(np.random.rand(5))',
                                         ret_type='np.float32',
                                         func='cublasSdot',
                                         check='np.allclose(d, np.dot(x, y))')

_libcublas.cublasDdot_v2.restype = int
_libcublas.cublasDdot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasDdot(handle, n, x, incx, y, incy):
    result = ctypes.c_double()
    status = _libcublas.cublasDdot_v2(handle, n,
                                      int(x), incx, int(y), incy,
                                      ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)

cublasDdot.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.float64(np.random.rand(5))',
                                         ret_type='np.float64',
                                         func='cublasDdot',
                                         check='np.allclose(d, np.dot(x, y))')

_libcublas.cublasCdotu_v2.restype = int
_libcublas.cublasCdotu_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasCdotu(handle, n, x, incx, y, incy):
    result = cuda.cuFloatComplex()
    status = _libcublas.cublasCdotu_v2(handle, n,
                                       int(x), incx, int(y), incy,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex64(result.value)

cublasCdotu.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type='np.complex64',
                                         func='cublasCdotu',
                                         check='np.allclose(d, np.dot(x, y))')

_libcublas.cublasCdotc_v2.restype = int
_libcublas.cublasCdotc_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasCdotc(handle, n, x, incx, y, incy):
    result = cuda.cuFloatComplex()
    status = _libcublas.cublasCdotc_v2(handle, n,
                                       int(x), incx, int(y), incy,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex64(result.value)

cublasCdotc.__doc__ = _DOT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type='np.complex64',
                                         func='cublasCdotc',
                                         check='np.allclose(d, np.dot(np.conj(x), y))')

_libcublas.cublasZdotu_v2.restype = int
_libcublas.cublasZdotu_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasZdotu(handle, n, x, incx, y, incy):
    result = cuda.cuDoubleComplex()
    status = _libcublas.cublasZdotu_v2(handle, n,
                                       int(x), incx, int(y), incy,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex128(result.value)

cublasZdotu.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                          real='complex',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          ret_type='np.complex128',
                                          func='cublasZdotu',
                                          check='np.allclose(d, np.dot(x, y))')

_libcublas.cublasZdotc_v2.restype = int
_libcublas.cublasZdotc_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasZdotc(handle, n, x, incx, y, incy):
    result = cuda.cuDoubleComplex()
    status = _libcublas.cublasZdotc_v2(handle, n,
                                       int(x), incx, int(y), incy,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex128(result.value)

cublasZdotc.__doc__ = _DOT_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         ret_type='np.complex128',
                                         func='cublasZdotc',
                                         check='np.allclose(d, np.dot(np.conj(x), y))')

# SNRM2, DNRM2, SCNRM2, DZNRM2
_NRM2_doc = Template(
"""
    Euclidean norm (2-norm) of real vector.

    Computes the Euclidean norm of a ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    nrm : ${ret_type}
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> nrm = ${func}(h, x.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True
    
""")

_libcublas.cublasSnrm2_v2.restype = int
_libcublas.cublasSnrm2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasSnrm2(handle, n, x, incx):
    result = ctypes.c_float()
    status = _libcublas.cublasSnrm2_v2(handle,
                                       n, int(x), incx,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)
    
cublasSnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.float32(np.random.rand(5))',
                                         ret_type = 'numpy.float32',
                                         func='cublasSnrm2')

_libcublas.cublasDnrm2_v2.restype = int
_libcublas.cublasDnrm2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasDnrm2(handle, n, x, incx):
    result = ctypes.c_double()
    status = _libcublas.cublasDnrm2_v2(handle,
                                       n, int(x), incx,
                                       ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)
    
cublasDnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.float64(np.random.rand(5))',
                                         ret_type = 'numpy.float64',
                                         func='cublasDnrm2')

_libcublas.cublasScnrm2_v2.restype = int
_libcublas.cublasScnrm2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasScnrm2(handle, n, x, incx):
    result = cuda.cuFloatComplex()
    status = _libcublas.cublasScnrm2_v2(handle,
                                        n, int(x), incx,
                                        ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex64(result.value)
    
cublasScnrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         ret_type = 'numpy.complex64',
                                         func='cublasScnrm2')

_libcublas.cublasDznrm2_v2.restype = int
_libcublas.cublasDznrm2_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p]
def cublasDznrm2(handle, n, x, incx):
    result = cuda.cuDoubleComplex()
    status = _libcublas.cublasDznrm2_v2(handle,
                                        n, int(x), incx,
                                        ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex128(result.value)
    
cublasDznrm2.__doc__ = \
                    _NRM2_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         ret_type = 'numpy.complex128',
                                         func='cublasDznrm2')


# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_ROT_doc = Template(
"""
    Apply a ${real} rotation to ${real} vectors (${precision})

    Multiplies the ${precision} matrix `[[c, s], [-s, c]]`
    with the 2 x `n` ${precision} matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    c : ${c_type}
        Element of rotation matrix.
    s : ${s_type}
        Element of rotation matrix.

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> s = ${s_val}; c = ${c_val};
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> ${func}(h, x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1, c, s)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), c*x+s*y)
    True
    >>> np.allclose(y_gpu.get(), -s*x+c*y)
    True
    
""")

_libcublas.cublasSrot_v2.restype = int
_libcublas.cublasSrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasSrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasSrot_v2(handle,
                                      n, int(x), incx,
                                      int(y), incy,
                                      ctypes.byref(ctypes.c_float(c)),
                                      ctypes.byref(ctypes.c_float(s)))

    cublasCheckStatus(status)
        
cublasSrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                         real='real',
                                         c_type='numpy.float32',
                                         s_type='numpy.float32',
                                         c_val='np.float32(np.random.rand())',
                                         s_val='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSrot')

_libcublas.cublasDrot_v2.restype = int
_libcublas.cublasDrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasDrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasDrot_v2(handle,
                                      n, int(x),
                                      incx, int(y), incy,
                                      ctypes.byref(ctypes.c_double(c)),
                                      ctypes.byref(ctypes.c_double(s)))
    cublasCheckStatus(status)
        
cublasDrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                         real='real',
                                         c_type='numpy.float64',
                                         s_type='numpy.float64',
                                         c_val='np.float64(np.random.rand())',
                                         s_val='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDrot')

_libcublas.cublasCrot_v2.restype = int
_libcublas.cublasCrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasCrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasCrot_v2(handle,
                                      n, int(x),
                                      incx, int(y), incy,
                                      ctypes.byref(ctypes.c_float(c)),
                                      ctypes.byref(cuda.cuFloatComplex(s.real,
                                                                       s.imag)))
    cublasCheckStatus(status)
        
cublasCrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                         real='complex',
                                         c_type='numpy.float32',
                                         s_type='numpy.complex64',
                                         c_val='np.float32(np.random.rand())',
                                         s_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCrot')

_libcublas.cublasCsrot_v2.restype = int
_libcublas.cublasCsrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasCsrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasCsrot_v2(handle,
                                       n, int(x),
                                       incx, int(y), incy,
                                       c, s)
    cublasCheckStatus(status)
        
cublasCsrot.__doc__ = _ROT_doc.substitute(precision='single-precision',
                                          real='complex',
                                          c_type='numpy.float32',
                                          s_type='numpy.float32',
                                          c_val='np.float32(np.random.rand())',
                                          s_val='np.float32(np.random.rand())',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                          func='cublasCsrot')

_libcublas.cublasZrot_v2.restype = int
_libcublas.cublasZrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasZrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasZrot_v2(handle,
                                      n, int(x),
                                      incx, int(y), incy,
                                      c,
                                      ctypes.byref(cuda.cuDoubleComplex(s.real, s.imag)))
    cublasCheckStatus(status)
        
cublasZrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                         real='complex',
                                         c_type='numpy.float64',
                                         s_type='numpy.complex128',
                                         c_val='np.float64(np.random.rand())',
                                         s_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZrot')

_libcublas.cublasZdrot_v2.restype = int
_libcublas.cublasZdrot_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p]
def cublasZdrot(handle, n, x, incx, y, incy, c, s):
    status = _libcublas.cublasZdrot_v2(handle,
                                       n, int(x),
                                       incx, int(y), incy,
                                       c, s)
    cublasCheckStatus(status)
        
cublasZdrot.__doc__ = _ROT_doc.substitute(precision='double-precision',
                                          real='complex',
                                          c_type='numpy.float64',
                                          s_type='numpy.float64',
                                          c_val='np.float64(np.random.rand())',
                                          s_val='np.float64(np.random.rand())',
                                          data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                          func='cublasZdrot')


# SROTG, DROTG, CROTG, ZROTG
_ROTG_doc = Template(
"""
    Construct a ${precision} ${real} Givens rotation matrix.

    Constructs the ${precision} ${real} Givens rotation matrix
    `G = [[c, s], [-s, c]]` such that
    `dot(G, [[a], [b]] == [[r], [0]]`, where
    `c**2+s**2 == 1` and `r == a**2+b**2` for real numbers and
    `c**2+(conj(s)*s) == 1` and `r ==
    (a/abs(a))*sqrt(abs(a)**2+abs(b)**2)` for `a != 0` and `r == b`
    for `a == 0`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    a, b : ${type}
        Entries of vector whose second entry should be zeroed
        out by the rotation.

    Returns
    -------
    r : ${type}
        Defined above.
    c : ${c_type}
        Cosine component of rotation matrix.
    s : ${s_type}
        Sine component of rotation matrix.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> a = ${a_val}
    >>> b = ${b_val}
    >>> h = cublasCreate()
    >>> r, c, s = ${func}(h, a, b)
    >>> cublasDestroy(h)
    >>> np.allclose(np.dot(np.array([[c, s], [-np.conj(s), c]]), np.array([[a], [b]])), np.array([[r], [0.0]]))
    True

""")

_libcublas.cublasSrotg_v2.restype = int
_libcublas.cublasSrotg_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cublasSrotg(handle, a, b):
    _a = ctypes.c_float(a)
    _b = ctypes.c_float(b)
    _c = ctypes.c_float()
    _s = ctypes.c_float()
    status = _libcublas.cublasSrotg_v2(handle,
                                       ctypes.byref(_a), ctypes.byref(_b),
                                       ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.float32(_a.value), np.float32(_c.value), np.float32(_s.value)
                                  
cublasSrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='single-precision',
                                         real='real',
                                         type='numpy.float32',
                                         c_type='numpy.float32',
                                         s_type='numpy.float32',
                                         a_val='np.float32(np.random.rand())',
                                         b_val='np.float32(np.random.rand())',
                                         func='cublasSrotg')

_libcublas.cublasDrotg_v2.restype = int
_libcublas.cublasDrotg_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cublasDrotg(handle, a, b):
    _a = ctypes.c_double(a)
    _b = ctypes.c_double(b)
    _c = ctypes.c_double()
    _s = ctypes.c_double()
    status = _libcublas.cublasDrotg_v2(handle,
                                       ctypes.byref(_a), ctypes.byref(_b),
                                       ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.float64(_a.value), np.float64(_c.value), np.float64(_s.value)
                                  
cublasDrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='double-precision',
                                         real='real',
                                         type='numpy.float64',
                                         c_type='numpy.float64',
                                         s_type='numpy.float64',
                                         a_val='np.float64(np.random.rand())',
                                         b_val='np.float64(np.random.rand())',
                                         func='cublasDrotg')

_libcublas.cublasCrotg_v2.restype = int
_libcublas.cublasCrotg_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cublasCrotg(handle, a, b):
    _a = cuda.cuFloatComplex(a.real, a.imag)
    _b = cuda.cuFloatComplex(b.real, b.imag)
    _c = ctypes.c_float()
    _s = cuda.cuFloatComplex()
    status = _libcublas.cublasCrotg_v2(handle,
                                       ctypes.byref(_a), _b,
                                       ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.complex64(_a.value), np.float32(_c.value), np.complex64(_s.value)
                                  
cublasCrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='single-precision',
                                         real='complex',
                                         type='numpy.complex64',
                                         c_type='numpy.float32',
                                         s_type='numpy.complex64',
                                         a_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         b_val='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         func='cublasCrotg')

_libcublas.cublasZrotg_v2.restype = int
_libcublas.cublasZrotg_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cublasZrotg(handle, a, b):
    _a = cuda.cuDoubleComplex(a.real, a.imag)
    _b = cuda.cuDoubleComplex(b.real, b.imag)
    _c = ctypes.c_double()
    _s = cuda.cuDoubleComplex()
    status = _libcublas.cublasZrotg_v2(handle,
                                       ctypes.byref(_a), _b,
                                       ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.complex128(_a.value), np.float64(_c.value), np.complex128(_s.value)
                                  
cublasZrotg.__doc__ = \
                    _ROTG_doc.substitute(precision='double-precision',
                                         real='complex',
                                         type='numpy.complex128',
                                         c_type='numpy.float64',
                                         s_type='numpy.complex128',
                                         a_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         b_val='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         func='cublasZrotg')

# SROTM, DROTM (need to add example)
_ROTM_doc = Template(        
"""
    Apply a ${precision} real modified Givens rotation.

    Applies the ${precision} real modified Givens rotation matrix `h`
    to the 2 x `n` matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} real input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} real input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sparam : numpy.ndarray
        sparam[0] contains the `flag` described below;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.

    Notes
    -----
    The rotation matrix may assume the following values:

    for `flag` == -1.0, `h` == `[[h00, h01], [h10, h11]]`
    for `flag` == 0.0,  `h` == `[[1.0, h01], [h10, 1.0]]`
    for `flag` == 1.0,  `h` == `[[h00, 1.0], [-1.0, h11]]`
    for `flag` == -2.0, `h` == `[[1.0, 0.0], [0.0, 1.0]]`

    Both `x` and `y` must contain `n` elements.
    
""")

_libcublas.cublasSrotm_v2.restype = int
_libcublas.cublasSrotm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasSrotm(handle, n, x, incx, y, incy, sparam):
    status = _libcublas.cublasSrotm_v2(handle,
                                       n, int(x), incx, int(y),
                                       incy, int(sparam.ctypes.data))
    cublasCheckStatus(status)

cublasSrotm.__doc__ = \
                    _ROTM_doc.substitute(precision='single-precision')

_libcublas.cublasDrotm_v2.restype = int
_libcublas.cublasDrotm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasDrotm(handle, n, x, incx, y, incy, sparam):
    status = _libcublas.cublasDrotm_v2(handle,
                                       n, int(x), incx, int(y),
                                       incy, int(sparam.ctypes.data))
    cublasCheckStatus(status)

cublasDrotm.__doc__ = \
                    _ROTM_doc.substitute(precision='double-precision')
                                        
# SROTMG, DROTMG (need to add example)
_ROTMG_doc = Template( 
"""
    Construct a ${precision} real modified Givens rotation matrix.

    Constructs the ${precision} real modified Givens rotation matrix
    `h = [[h11, h12], [h21, h22]]` that zeros out the second entry of
    the vector `[[sqrt(d1)*x1], [sqrt(d2)*x2]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    d1 : ${type}
        ${precision} real value.
    d2 : ${type}
        ${precision} real value.
    x1 : ${type}
        ${precision} real value.
    x2 : ${type}
        ${precision} real value.

    Returns
    -------
    sparam : numpy.ndarray
        sparam[0] contains the `flag` described below;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.
        
    Notes
    -----
    The rotation matrix may assume the following values:

    for `flag` == -1.0, `h` == `[[h00, h01], [h10, h11]]`
    for `flag` == 0.0,  `h` == `[[1.0, h01], [h10, 1.0]]`
    for `flag` == 1.0,  `h` == `[[h00, 1.0], [-1.0, h11]]`
    for `flag` == -2.0, `h` == `[[1.0, 0.0], [0.0, 1.0]]`

""")

_libcublas.cublasSrotmg_v2.restype = int
_libcublas.cublasSrotmg_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def cublasSrotmg(handle, d1, d2, x1, y1):
    _d1 = ctypes.c_float(d1)
    _d2 = ctypes.c_float(d2)
    _x1 = ctypes.c_float(x1)
    _y1 = ctypes.c_float(y1)
    sparam = np.empty(5, np.float32)

    status = _libcublas.cublasSrotmg_v2(handle,
                                        ctypes.byref(_d1), ctypes.byref(_d2),
                                        ctypes.byref(_x1), ctypes.byref(_y1),
                                        int(sparam.ctypes.data))
    cublasCheckStatus(status)        
    return sparam

cublasSrotmg.__doc__ = \
                     _ROTMG_doc.substitute(precision='single-precision',
                                           type='numpy.float32')

_libcublas.cublasDrotmg_v2.restype = int
_libcublas.cublasDrotmg_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p]
def cublasDrotmg(handle, d1, d2, x1, y1):
    _d1 = ctypes.c_double(d1)
    _d2 = ctypes.c_double(d2)
    _x1 = ctypes.c_double(x1)
    _y1 = ctypes.c_double(y1)
    sparam = np.empty(5, np.float64)

    status = _libcublas.cublasDrotmg_v2(handle,
                                        ctypes.byref(_d1), ctypes.byref(_d2),
                                        ctypes.byref(_x1), ctypes.byref(_y1),
                                        int(sparam.ctypes.data))
    cublasCheckStatus(status)        
    return sparam

cublasDrotmg.__doc__ = \
                     _ROTMG_doc.substitute(precision='double-precision',
                                           type='numpy.float64')

# SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_SCAL_doc = Template(
"""
    Scale a ${precision} ${real} vector by a ${precision} ${a_real} scalar.

    Replaces a ${precision} ${real} vector `x` with
    `alpha * x`.
    
    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = ${alpha}
    >>> h = cublasCreate()
    >>> ${func}(h, x.size, alpha, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True
    
""")

_libcublas.cublasSscal_v2.restype = int
_libcublas.cublasSscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasSscal_v2(handle, n,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasSscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='real',
                                         a_real='real',
                                         a_type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSscal')

_libcublas.cublasDscal_v2.restype = int
_libcublas.cublasDscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasDscal_v2(handle, n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasDscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='real',
                                         a_real='real',
                                         a_type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDscal')

_libcublas.cublasCscal_v2.restype = int
_libcublas.cublasCscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasCscal_v2(handle, n,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasCscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='complex',
                                         a_real='complex',
                                         a_type='numpy.complex64',
                                         alpha='np.complex64(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCscal')

_libcublas.cublasCsscal_v2.restype = int
_libcublas.cublasCsscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCsscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasCsscal_v2(handle, n,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasCsscal.__doc__ = \
                    _SCAL_doc.substitute(precision='single-precision',
                                         real='complex',
                                         a_real='real',
                                         a_type='numpy.float32',
                                         alpha='np.float32(np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCsscal')

_libcublas.cublasZscal_v2.restype = int
_libcublas.cublasZscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasZscal_v2(handle, n,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasZscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='complex',
                                         a_real='complex',
                                         a_type='numpy.complex128',
                                         alpha='np.complex128(np.random.rand()+1j*np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZscal')

_libcublas.cublasZdscal_v2.restype = int
_libcublas.cublasZdscal_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZdscal(handle, n, alpha, x, incx):
    status = _libcublas.cublasZdscal_v2(handle, n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(x), incx)
    cublasCheckStatus(status)
        
cublasZdscal.__doc__ = \
                    _SCAL_doc.substitute(precision='double-precision',
                                         real='complex',
                                         a_real='real',
                                         a_type='numpy.float64',
                                         alpha='np.float64(np.random.rand())',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZdscal')

# SSWAP, DSWAP, CSWAP, ZSWAP
_SWAP_doc = Template(
"""
    Swap ${precision} ${real} vectors.

    Swaps the contents of one ${precision} ${real} vector with those
    of another ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to ${precision} ${real} input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate() 
    >>> ${func}(x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), y)
    True
    >>> np.allclose(y_gpu.get(), x)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

""")

_libcublas.cublasSswap_v2.restype = int
_libcublas.cublasSswap_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSswap(handle, n, x, incx, y, incy):
    status = _libcublas.cublasSswap_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasSswap.__doc__ = \
                    _SWAP_doc.substitute(precision='single-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float32)',
                                         func='cublasSswap')

_libcublas.cublasDswap_v2.restype = int
_libcublas.cublasDswap_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]    
def cublasDswap(handle, n, x, incx, y, incy):
    status = _libcublas.cublasDswap_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasDswap.__doc__ = \
                    _SWAP_doc.substitute(precision='double-precision',
                                         real='real',
                                         data='np.random.rand(5).astype(np.float64)',
                                         func='cublasDswap')

_libcublas.cublasCswap_v2.restype = int
_libcublas.cublasCswap_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCswap(handle, n, x, incx, y, incy):
    status = _libcublas.cublasCswap_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasCswap.__doc__ = \
                    _SWAP_doc.substitute(precision='single-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)',
                                         func='cublasCswap')

_libcublas.cublasZswap_v2.restype = int
_libcublas.cublasZswap_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZswap(handle, n, x, incx, y, incy):
    status = _libcublas.cublasZswap_v2(handle,
                                       n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)

cublasZswap.__doc__ = \
                    _SWAP_doc.substitute(precision='double-precision',
                                         real='complex',
                                         data='(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)',
                                         func='cublasZswap')

### BLAS Level 2 Functions ###

# SGBMV, DGVMV, CGBMV, ZGBMV 
_libcublas.cublasSgbmv_v2.restype = int
_libcublas.cublasSgbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda,
                x, incx, beta, y, incy):
    """
    Matrix-vector product for real general banded matrix.

    """

    status = _libcublas.cublasSgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda,
                                       int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasDgbmv_v2.restype = int
_libcublas.cublasDgbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, 
                x, incx, beta, y, incy):
    """
    Matrix-vector product for real general banded matrix.

    """

    status = _libcublas.cublasDgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasCgbmv_v2.restype = int
_libcublas.cublasCgbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda,
                x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general banded matrix.

    """

    status = _libcublas.cublasCgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasZgbmv_v2.restype = int
_libcublas.cublasZgbmv_v2.argtypes = [ctypes.c_char,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, 
                x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general banded matrix.

    """

    status = _libcublas.cublasZgbmv_v2(handle,
                                       trans, m, n, kl, ku,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                              int(y), incy)
    cublasCheckStatus(status)
    
# SGEMV, DGEMV, CGEMV, ZGEMV # XXX need to adjust
# _GEMV_doc = Template( 
# """
#     Matrix-vector product for ${precision} ${real} general matrix.

#     Computes the product `alpha*op(A)*x+beta*y`, where `op(A)` == `A`
#     or `op(A)` == `A.T`, and stores it in `y`.

#     Parameters
#     ----------
#     trans : char
#         If `upper(trans)` in `['T', 'C']`, assume that `A` is
#         transposed.
#     m : int
#         Number of rows in `A`.
#     n : int
#         Number of columns in `A`.
#     alpha : ${a_type}
#         `A` is multiplied by this quantity. 
#     A : ctypes.c_void_p
#         Pointer to ${precision} matrix. The matrix has
#         shape `(lda, n)` if `upper(trans)` == 'N', `(lda, m)`
#         otherwise.
#     lda : int
#         Leading dimension of `A`.
#     X : ctypes.c_void_p
#         Pointer to ${precision} array of length at least
#         `(1+(n-1)*abs(incx))` if `upper(trans) == 'N',
#         `(1+(m+1)*abs(incx))` otherwise.
#     incx : int
#         Spacing between elements of `x`. Must be nonzero.
#     beta : ${a_type}
#         `y` is multiplied by this quantity. If zero, `y` is ignored.
#     y : ctypes.c_void_p
#         Pointer to ${precision} array of length at least
#         `(1+(m+1)*abs(incy))` if `upper(trans)` == `N`,
#         `(1+(n+1)*abs(incy))` otherwise.
#     incy : int
#         Spacing between elements of `y`. Must be nonzero.

#     Examples
#     --------
#     >>> import pycuda.autoinit
#     >>> import pycuda.gpuarray as gpuarray
#     >>> import numpy as np
#     >>> a = np.random.rand(2, 3).astype(np.float32)
#     >>> x = np.random.rand(3, 1).astype(np.float32)
#     >>> a_gpu = gpuarray.to_gpu(a.T.copy())
#     >>> x_gpu = gpuarray.to_gpu(x)
#     >>> y_gpu = gpuarray.empty((2, 1), np.float32)
#     >>> alpha = np.float32(1.0)
#     >>> beta = np.float32(0)
#     >>> h = cublasCreate()
#     >>> ${func}(h, 'n', 2, 3, alpha, a_gpu.gpudata, 2, x_gpu.gpudata, 1, beta, y_gpu.gpudata, 1)
#     >>> cublasDestroy(h)
#     >>> np.allclose(y_gpu.get(), np.dot(a, x))
#     True

# """
    
_libcublas.cublasSgemv_v2.restype = int
_libcublas.cublasSgemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """

    status = _libcublas.cublasSgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(ctypes.c_float(alpha)), int(A), lda,
                                       int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)), int(y), incy) 
    cublasCheckStatus(status)
        
_libcublas.cublasDgemv_v2.restype = int
_libcublas.cublasDgemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real general matrix.

    """

    status = _libcublas.cublasDgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)
    
_libcublas.cublasCgemv_v2.restype = int
_libcublas.cublasCgemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """

    status = _libcublas.cublasCgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)
    
_libcublas.cublasZgemv_v2.restype = int
_libcublas.cublasZgemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,        
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex general matrix.

    """

    status = _libcublas.cublasZgemv_v2(handle,
                                       _CUBLAS_OP[trans], m, n,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
_libcublas.cublasSger_v2.restype = int
_libcublas.cublasSger_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on real general matrix.

    """
    
    status = _libcublas.cublasSger_v2(handle,
                                      m, n,
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx,
                                      int(y), incy, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasDger_v2.restype = int
_libcublas.cublasDger_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on real general matrix.

    """
    
    status = _libcublas.cublasDger_v2(handle,
                                      m, n,
                                      ctypes.byref(ctypes.c_double(alpha)),
                                      int(x), incx,
                                      int(y), incy, int(A), lda)
    cublasCheckStatus(status)
    
_libcublas.cublasCgerc_v2.restype = int
_libcublas.cublasCgerc_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasCgerc_v2(handle,
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                            alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)
    
_libcublas.cublasCgeru_v2.restype = int
_libcublas.cublasCgeru_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasCgeru_v2(handle,
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                              alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)
    
_libcublas.cublasZgerc_v2.restype = int
_libcublas.cublasZgerc_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasZgerc_v2(handle,
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                               alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasZgeru_v2.restype = int
_libcublas.cublasZgeru_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-1 operation on complex general matrix.

    """

    status = _libcublas.cublasZgeru_v2(handle,
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                               alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)

# SSBMV, DSBMV 
_libcublas.cublasSsbmv_v2.restype = int
_libcublas.cublasSsbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]

def cublasSsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-banded matrix.

    """

    status = _libcublas.cublasSsbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], n, k,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)
        
_libcublas.cublasDsbmv_v2.restype = int
_libcublas.cublasDsbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-banded matrix.

    """

    status = _libcublas.cublasDsbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], n, k,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(y), incy)
    cublasCheckStatus(status)
        
# SSPMV, DSPMV
_libcublas.cublasSspmv_v2.restype = int
_libcublas.cublasSspmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """

    status = _libcublas.cublasSspmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       ctypes.byref(ctypes.c_float(AP)),
                                       int(x),
                                       incx,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(y),
                                       incy)
    cublasCheckStatus(status)
        
_libcublas.cublasDspmv_v2.restype = int
_libcublas.cublasDspmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric-packed matrix.

    """

    status = _libcublas.cublasDspmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n,
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       ctypes.byref(ctypes.c_double(AP)),
                                       int(x),
                                       incx,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(y),
                                       incy)
    cublasCheckStatus(status)

# SSPR, DSPR
_libcublas.cublasSspr_v2.restype = int
_libcublas.cublasSspr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasSspr(handle, uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on real symmetric-packed matrix.

    """
    
    status = _libcublas.cublasSspr_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], n,                                       
                                      ctypes.byref(ctypes.c_float(alpha)), 
                                      int(x), incx, int(AP))                                      
    cublasCheckStatus(status)


_libcublas.cublasDspr_v2.restype = int
_libcublas.cublasDspr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasDspr(handle, uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on real symmetric-packed matrix.

    """

    status = _libcublas.cublasDspr_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], n,                                       
                                      ctypes.byref(ctypes.c_double(alpha)), 
                                      int(x), incx, int(AP))                                           
    cublasCheckStatus(status)

# SSPR2, DSPR2
_libcublas.cublasSspr2_v2.restype = int
_libcublas.cublasSspr2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasSspr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """
    Rank-2 operation on real symmetric-packed matrix.

    """

    status = _libcublas.cublasSspr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx, int(y), incy, int(AP))    
                                                                              
    cublasCheckStatus(status)

_libcublas.cublasDspr2_v2.restype = int
_libcublas.cublasDspr2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasDspr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """
    Rank-2 operation on real symmetric-packed matrix.

    """

    status = _libcublas.cublasDspr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_double(alpha)), 
                                       int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)

# SSYMV, DSYMV, CSYMV, ZSYMV
_libcublas.cublasSsymv_v2.restype = int
_libcublas.cublasSsymv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """
    
    status = _libcublas.cublasSsymv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasDsymv_v2.restype = int
_libcublas.cublasDsymv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real symmetric matrix.
    
    """

    status = _libcublas.cublasDsymv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_double(alpha)), 
                                       int(A), lda, int(x), incx, 
                                       ctypes.byref(ctypes.c_double(beta)), 
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasCsymv_v2.restype = int
_libcublas.cublasCsymv_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex symmetric matrix.

    """

    status = _libcublas.cublasCsymv_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)), 
                                        int(A), lda, int(x), incx, 
                                        ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)), 
                                        int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasZsymv_v2.restype = int
_libcublas.cublasZsymv_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasZsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex symmetric matrix.

    """

    status = _libcublas.cublasZsymv_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)), 
                                        int(A), lda, int(x), incx, 
                                        ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                            beta.imag)), 
                                        int(y), incy)
    cublasCheckStatus(status)

# SSYR, DSYR, CSYR, ZSYR
_libcublas.cublasSsyr_v2.restype = int
_libcublas.cublasSsyr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasSsyr(handle, uplo, n, alpha, x, incx, A, lda): 
    """
    Rank-1 operation on real symmetric matrix.

    """
   
    status = _libcublas.cublasSsyr_v2(handle,
                                      _CUBLAS_FILL_MODE[uplo], n, 
                                      ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasDsyr_v2.restype = int
_libcublas.cublasDsyr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasDsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on real symmetric matrix.

    """

    status = _libcublas.cublasDsyr_v2(handle,
                                      _CUBLAS_FILL_MODE[uplo], n, 
                                      ctypes.byref(ctypes.c_double(alpha)), 
                                      int(x), incx, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasCsyr_v2.restype = int
_libcublas.cublasCsyr_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasCsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on complex symmetric matrix.

    """

    status = _libcublas.cublasCsyr_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                        int(x), incx, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasZsyr_v2.restype = int
_libcublas.cublasZsyr_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasZsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on complex symmetric matrix.

    """

    status = _libcublas.cublasZsyr_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                        alpha.imag)),
                                        int(x), incx, int(A), lda)
    cublasCheckStatus(status)

# SSYR2, DSYR2, CSYR2, ZSYR2
_libcublas.cublasSsyr2_v2.restype = int
_libcublas.cublasSsyr2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on real symmetric matrix.

    """

    status = _libcublas.cublasSsyr2_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(x), incx, int(y), incy,
                                       int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasDsyr2_v2.restype = int
_libcublas.cublasDsyr2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                   
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on real symmetric matrix.

    """

    status = _libcublas.cublasDsyr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], n, 
                                       ctypes.byref(ctypes.c_double(alpha)), 
                                       int(x), incx, int(y), incy, 
                                       int(A), lda)                                       
    cublasCheckStatus(status)

_libcublas.cublasCsyr2_v2.restype = int
_libcublas.cublasCsyr2_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,                                   
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on complex symmetric matrix.

    """

    status = _libcublas.cublasCsyr2_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)), 
                                        int(x), incx, int(y), incy, 
                                        int(A), lda)                                       
    cublasCheckStatus(status)
 
_libcublas.cublasZsyr2_v2.restype = int
_libcublas.cublasZsyr2_v2.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,                                   
                                        ctypes.c_void_p,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int,
                                        ctypes.c_void_p,
                                        ctypes.c_int]
def cublasZsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on complex symmetric matrix.

    """

    status = _libcublas.cublasZsyr2_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], n, 
                                        ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)), 
                                        int(x), incx, int(y), incy, 
                                        int(A), lda)                                       
    cublasCheckStatus(status)

# STBMV, DTBMV, CTBMV, ZTBMV
_libcublas.cublasStbmv_v2.restype = int
_libcublas.cublasStbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for real triangular-banded matrix.

    """
    
    status = _libcublas.cublasStbmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasDtbmv_v2.restype = int
_libcublas.cublasDtbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for real triangular-banded matrix.

    """

    status = _libcublas.cublasDtbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)                                       
    cublasCheckStatus(status)

_libcublas.cublasCtbmv_v2.restype = int
_libcublas.cublasCtbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular-banded matrix.

    """
    
    status = _libcublas.cublasCtbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)                           
    cublasCheckStatus(status)

_libcublas.cublasZtbmv_v2.restype = int
_libcublas.cublasZtbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular-banded matrix.

    """
    
    status = _libcublas.cublasZtbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

# STBSV, DTBSV, CTBSV, ZTBSV
_libcublas.cublasStbsv_v2.restype = int
_libcublas.cublasStbsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve real triangular-banded system with one right-hand side.

    """
    
    status = _libcublas.cublasStbsv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)                                       
    cublasCheckStatus(status)

_libcublas.cublasDtbsv_v2.restype = int
_libcublas.cublasDtbsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve real triangular-banded system with one right-hand side.

    """

    status = _libcublas.cublasDtbsv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)                           
    cublasCheckStatus(status)

_libcublas.cublasCtbsv_v2.restype = int
_libcublas.cublasCtbsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve complex triangular-banded system with one right-hand side.

    """
    
    status = _libcublas.cublasCtbsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)                                       
    cublasCheckStatus(status)

_libcublas.cublasZtbsv_v2.restype = int
_libcublas.cublasZtbsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """
    Solve complex triangular-banded system with one right-hand side.

    """
    
    status = _libcublas.cublasZtbsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

# STPMV, DTPMV, CTPMV, ZTPMV
_libcublas.cublasStpmv_v2.restype = int
_libcublas.cublasStpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for real triangular-packed matrix.

    """
    
    status = _libcublas.cublasStpmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasCtpmv_v2.restype = int
_libcublas.cublasCtpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for complex triangular-packed matrix.

    """
    
    status = _libcublas.cublasCtpmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasDtpmv_v2.restype = int
_libcublas.cublasDtpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for real triangular-packed matrix.

    """

    status = _libcublas.cublasDtpmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasZtpmv_v2.restype = int
_libcublas.cublasZtpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Matrix-vector product for complex triangular-packed matrix.

    """
    
    status = _libcublas.cublasZtpmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

# STPSV, DTPSV, CTPSV, ZTPSV
_libcublas.cublasStpsv_v2.restype = int
_libcublas.cublasStpsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve real triangular-packed system with one right-hand side.

    """
    
    status = _libcublas.cublasStpsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)


_libcublas.cublasDtpsv_v2.restype = int
_libcublas.cublasDtpsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve real triangular-packed system with one right-hand side.

    """

    status = _libcublas.cublasDtpsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasCtpsv_v2.restype = int
_libcublas.cublasCtpsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve complex triangular-packed system with one right-hand side.
    
    """
    
    status = _libcublas.cublasCtpsv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasZtpsv_v2.restype = int
_libcublas.cublasZtpsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """
    Solve complex triangular-packed system with one right-hand size.

    """
    
    status = _libcublas.cublasZtpsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(AP), int(x), incx)
    cublasCheckStatus(status)

# STRMV, DTRMV, CTRMV, ZTRMV
_libcublas.cublasStrmv_v2.restype = int
_libcublas.cublasStrmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStrmv(handle, uplo, trans, diag, n, A, lda, x, inx):
    """
    Matrix-vector product for real triangular matrix.

    """
    
    status = _libcublas.cublasStrmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), inx)                                       
    cublasCheckStatus(status)

_libcublas.cublasCtrmv_v2.restype = int
_libcublas.cublasCtrmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular matrix.

    """
    
    status = _libcublas.cublasCtrmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasDtrmv_v2.restype = int
_libcublas.cublasDtrmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, inx):
    """
    Matrix-vector product for real triangular matrix.

    """

    status = _libcublas.cublasDtrmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), inx)
    cublasCheckStatus(status)

_libcublas.cublasZtrmv_v2.restype = int
_libcublas.cublasZtrmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtrmv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Matrix-vector product for complex triangular matrix.

    """
    
    status = _libcublas.cublasZtrmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

# STRSV, DTRSV, CTRSV, ZTRSV
_libcublas.cublasStrsv_v2.restype = int
_libcublas.cublasStrsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve real triangular system with one right-hand side.

    """
    
    status = _libcublas.cublasStrsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)                                       
    cublasCheckStatus(status)

_libcublas.cublasDtrsv_v2.restype = int
_libcublas.cublasDtrsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve real triangular system with one right-hand side.

    """

    status = _libcublas.cublasDtrsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasCtrsv_v2.restype = int
_libcublas.cublasCtrsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve complex triangular system with one right-hand side.

    """
    
    status = _libcublas.cublasCtrsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

_libcublas.cublasZtrsv_v2.restype = int
_libcublas.cublasZtrsv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """
    Solve complex triangular system with one right-hand side.

    """
    
    status = _libcublas.cublasZtrsv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)

# CHEMV, ZHEMV
_libcublas.cublasChemv_v2.restype = int
_libcublas.cublasChemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix vector product for Hermitian matrix.
    
    """
    
    status = _libcublas.cublasChemv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                           alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasZhemv_v2.restype = int
_libcublas.cublasZhemv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                       
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian matrix.

    """
    
    status = _libcublas.cublasZhemv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real, 
                                                                         beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

# CHBMV, ZHBMV
_libcublas.cublasChbmv_v2.restype = int
_libcublas.cublasChbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-banded matrix.

    """
    
    status = _libcublas.cublasChbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, k,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasZhbmv_v2.restype = int
_libcublas.cublasZhbmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian banded matrix.

    """
    
    status = _libcublas.cublasZhbmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, k,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real, 
                                                                         beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

# CHPMV, ZHPMV
_libcublas.cublasChpmv_v2.restype = int
_libcublas.cublasChpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasChpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-packed matrix.

    """
    
    status = _libcublas.cublasChpmv_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                           alpha.imag)),
                                       int(AP), int(x), incx,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

_libcublas.cublasZhpmv_v2.restype = int
_libcublas.cublasZhpmv_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """
    Matrix-vector product for Hermitian-packed matrix.

    """
    
    status = _libcublas.cublasZhpmv_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)),
                                       int(AP), int(x), incx,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real, 
                                                                         beta.imag)),
                                       int(y), incy)
    cublasCheckStatus(status)

# CHER, ZHER
_libcublas.cublasCher_v2.restype = int
_libcublas.cublasCher_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasCher(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on Hermitian matrix.

    """

    status = _libcublas.cublasCher_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, alpha, int(x), incx, int(A), lda)
    cublasCheckStatus(status)

_libcublas.cublasZher_v2.restype = int
_libcublas.cublasZher_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,                                     
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p,
                                     ctypes.c_int]
def cublasZher(handle, uplo, n, alpha, x, incx, A, lda):
    """
    Rank-1 operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasZher_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, alpha, int(x), incx, int(A), lda)
    cublasCheckStatus(status)


# CHER2, ZHER2
_libcublas.cublasCher2_v2.restype = int
_libcublas.cublasCher2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on Hermitian matrix.


    """
    
    status = _libcublas.cublasCher2_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                           alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)                           
    cublasCheckStatus(status)

_libcublas.cublasZher2_v2.restype = int
_libcublas.cublasZher2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """
    Rank-2 operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasZher2_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                            alpha.imag)),
                                       int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)

# CHPR, ZHPR
_libcublas.cublasChpr_v2.restype = int
_libcublas.cublasChpr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,                                     
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasChpr(handle, uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on Hermitian-packed matrix.
    
    """
    
    status = _libcublas.cublasChpr_v2(handle, 
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, ctypes.byref(ctypes.c_float(alpha)),
                                      int(x), incx, int(AP))
    cublasCheckStatus(status)

_libcublas.cublasZhpr_v2.restype = int
_libcublas.cublasZhpr_v2.argtypes = [ctypes.c_int,
                                     ctypes.c_int,
                                     ctypes.c_int,                                     
                                     ctypes.c_void_p,
                                     ctypes.c_void_p,
                                     ctypes.c_int,
                                     ctypes.c_void_p]
def cublasZhpr(handle, uplo, n, alpha, x, incx, AP):
    """
    Rank-1 operation on Hermitian-packed matrix.

    """
    
    status = _libcublas.cublasZhpr_v2(handle,
                                      _CUBLAS_FILL_MODE[uplo], 
                                      n, ctypes.byref(ctypes.c_double(alpha)),
                                      int(x), incx, int(AP))
    cublasCheckStatus(status)

# CHPR2, ZHPR2
_libcublas.cublasChpr2.restype = int
_libcublas.cublasChpr2.argtypes = [ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,                                   
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p]
def cublasChpr2(handle, uplo, n, alpha, x, inx, y, incy, AP):
    """
    Rank-2 operation on Hermitian-packed matrix.
    
    """

    status = _libcublas.cublasChpr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                           alpha.imag)),
                                       int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)

_libcublas.cublasZhpr2_v2.restype = int
_libcublas.cublasZhpr2_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p]
def cublasZhpr2(handle, uplo, n, alpha, x, inx, y, incy, AP):
    """
    Rank-2 operation on Hermitian-packed matrix.

    """
    
    status = _libcublas.cublasZhpr2_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,  
                                                                            alpha.imag)),
                                       int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)

# SGEMM, CGEMM, DGEMM, ZGEMM
_libcublas.cublasSgemm_v2.restype = int
_libcublas.cublasSgemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real general matrix.

    """

    status = _libcublas.cublasSgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasCgemm_v2.restype = int
_libcublas.cublasCgemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """

    status = _libcublas.cublasCgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasDgemm_v2.restype = int
_libcublas.cublasDgemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real general matrix.

    """

    status = _libcublas.cublasDgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZgemm_v2.restype = int
_libcublas.cublasZgemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """

    status = _libcublas.cublasZgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k, 
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)
    
# SSYMM, DSYMM, CSYMM, ZSYMM
_libcublas.cublasSsymm_v2.restype = int
_libcublas.cublasSsymm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for symmetric matrix.

    """
    
    status = _libcublas.cublasSsymm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb, 
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasDsymm_v2.restype = int
_libcublas.cublasDsymm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]

def cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real symmetric matrix.

    """
    
    status = _libcublas.cublasDsymm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo],
                                       m, n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb, 
                                       ctypes.byref(ctypes.c_double(beta)), 
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasCsymm_v2.restype = int
_libcublas.cublasCsymm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex symmetric matrix.

    """
    
    status = _libcublas.cublasCsymm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,                   
                                                                              alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real, 
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZsymm_v2.restype = int
_libcublas.cublasZsymm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):                
    """
    Matrix-matrix product for complex symmetric matrix.

    """
    
    status = _libcublas.cublasZsymm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], m, n,
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

# SSYRK, DSYRK, CSYRK, ZSYRK
_libcublas.cublasSsyrk_v2.restype = int
_libcublas.cublasSsyrk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasSsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, 
                                       ctypes.byref(ctypes.c_float(beta)), 
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasDsyrk_v2.restype = int
_libcublas.cublasDsyrk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasDsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,     
                                                                              alpha.imag)),
                                       int(A), lda, 
                                       ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasCsyrk_v2.restype = int
_libcublas.cublasCsyrk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on complex symmetric matrix.

    """
    
    status = _libcublas.cublasCsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,       
                                                                              alpha.imag)),
                                       int(A), lda,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real, 
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZsyrk_v2.restype = int
_libcublas.cublasZsyrk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on complex symmetric matrix.

    """
    
    status = _libcublas.cublasZsyrk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(cuda.cuDoubleComplex(alpha.real,    
                                                                               alpha.imag)),
                                       int(A), lda,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

# SSYR2K, DSYR2K, CSYR2K, ZSYR2K
_libcublas.cublasSsyr2k_v2.restype = int
_libcublas.cublasSsyr2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,                                       
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """
    
    status = _libcublas.cublasSsyr2k_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(ctypes.c_float(alpha)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(ctypes.c_float(beta)), 
                                        int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasDsyr2k_v2.restype = int
_libcublas.cublasDsyr2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,                                       
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasDsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on real symmetric matrix.

    """

    status = _libcublas.cublasDsyr2k_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(ctypes.c_double(alpha)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(ctypes.c_double(beta)), 
                                        int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasCsyr2k_v2.restype = int
_libcublas.cublasCsyr2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,                                       
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on complex symmetric matrix.

    """

    status = _libcublas.cublasCsyr2k_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,                
                                                                               alpha.imag)),
                                        int(A), lda, int(B), ldb,
                                        ctypes.byref(cuda.cuFloatComplex(beta.real, 
                                                                         beta.imag)),
                                        int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZsyr2k_v2.restype = int
_libcublas.cublasZsyr2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,                                       
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on complex symmetric matrix.
    
    """
    
    status = _libcublas.cublasZsyr2k_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                                alpha.imag)),
                                        int(A), lda, int(B), ldb,
                                        ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                          beta.imag)),
                                        int(C), ldc)
    cublasCheckStatus(status)

# STRMM, DTRMM, CTRMM, ZTRMM
_libcublas.cublasStrmm_v2.restype = int
_libcublas.cublasStrmm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,                                      
                                      ctypes.c_int]
def cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """
    Matrix-matrix product for real triangular matrix.

    """
    
    status = _libcublas.cublasStrmm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasDtrmm_v2.restype = int
_libcublas.cublasDtrmm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                       
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):                
    """
    Matrix-matrix product for real triangular matrix.

    """
    
    status = _libcublas.cublasDtrmm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasCtrmm_v2.restype = int
_libcublas.cublasCtrmm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                       
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):    
    """
    Matrix-matrix product for complex triangular matrix.

    """
    
    status = _libcublas.cublasCtrmm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                              alpha.imag)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)

_libcublas.cublasZtrmm_v2.restype = int
_libcublas.cublasZtrmm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """
    Matrix-matrix product for complex triangular matrix.

    """
    
    status = _libcublas.cublasZtrmm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,     
                                                                               alpha.imag)),
                                       int(A), lda, int(B), ldb, int(C), ldc)
    cublasCheckStatus(status)

# STRSM, DTRSM, CTRSM, ZTRSM
_libcublas.cublasStrsm_v2.restype = int
_libcublas.cublasStrsm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real triangular system with multiple right-hand sides.

    """
    
    status = _libcublas.cublasStrsm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)

_libcublas.cublasDtrsm_v2.restype = int
_libcublas.cublasDtrsm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a real triangular system with multiple right-hand sides.

    """
    
    status = _libcublas.cublasDtrsm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)

_libcublas.cublasCtrsm_v2.restype = int
_libcublas.cublasCtrsm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve a complex triangular system with multiple right-hand sides.

    """
    
    status = _libcublas.cublasCtrsm_v2(handle,
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                              alpha.imag)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)

_libcublas.cublasZtrsm_v2.restype = int
_libcublas.cublasZtrsm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZtrsm(handle, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb):
    """
    Solve complex triangular system with multiple right-hand sides.

    """
    
    status = _libcublas.cublasZtrsm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       _CUBLAS_DIAG[diag], 
                                       m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real,                    
                                                                               alpha.imag)),
                                       int(A), lda, int(B), ldb)
    cublasCheckStatus(status)

# CHEMM, ZHEMM
_libcublas.cublasChemm_v2.restype = int
_libcublas.cublasChemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex Hermitian matrix.

    """
    
    status = _libcublas.cublasChemm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], m, n,
                                       ctypes.byref(cuda.cuFloatComplex(alpha.real,
                                                                        alpha.imag)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuFloatComplex(beta.real, 
                                                                        beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZhemm_v2.restype = int
_libcublas.cublasZhemm_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZhemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):                
    """
    Matrix-matrix product for Hermitian matrix.

    """
    
    status = _libcublas.cublasZhemm_v2(handle, 
                                       _CUBLAS_SIDE_MODE[side], 
                                       _CUBLAS_FILL_MODE[uplo], m, n,                                       
                                       ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                         alpha.imag)),
                                                                         int(A), lda, int(B), ldb,
                                       ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                         beta.imag)),
                                       int(C), ldc)
    cublasCheckStatus(status)

# CHERK, ZHERK
_libcublas.cublasCherk_v2.restype = int
_libcublas.cublasCherk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasCherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasCherk_v2(handle, 
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans], 
                                       n, k, ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, 
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)

_libcublas.cublasZherk_v2.restype = int
_libcublas.cublasZherk_v2.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,                                      
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
def cublasZherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """
    Rank-k operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasZherk_v2(handle,
                                       _CUBLAS_FILL_MODE[uplo], 
                                       _CUBLAS_OP[trans],
                                       n, k, ctypes.byref(ctypes.c_double(alpha)),
                                       int(A), lda, 
                                       ctypes.byref(ctypes.c_double(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)

# CHER2K, ZHER2K
_libcublas.cublasCher2k_v2.restype = int
_libcublas.cublasCher2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_float,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasCher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on Hermitian matrix.

    """
    
    status = _libcublas.cublasCher2k_v2(handle, 
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(cuda.cuFloatComplex(alpha.real,                 
                                                                               alpha.imag)),
                                        int(A), lda, int(B), ldb, 
                                        ctypes.byref(cuda.cuFloatComplex(beta.real,
                                                                         beta.imag)),
                                        int(C), ldc)
    cublasCheckStatus(status)
        
_libcublas.cublasZher2k_v2.restype = int
_libcublas.cublasZher2k_v2.argtypes = [ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,                                       
                                       ctypes.c_void_p,                                       
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       ctypes.c_void_p,
                                       ctypes.c_int]
def cublasZher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Rank-2k operation on Hermitian matrix.

    """

    status = _libcublas.cublasZher2k_v2(handle,
                                        _CUBLAS_FILL_MODE[uplo], 
                                        _CUBLAS_OP[trans], 
                                        n, k, ctypes.byref(cuda.cuDoubleComplex(alpha.real,
                                                                                alpha.imag)),
                                        int(A), lda, int(B), ldb,
                                        ctypes.byref(cuda.cuDoubleComplex(beta.real,
                                                                          beta.imag)), 
                                        int(C), ldc)
    cublasCheckStatus(status)

### BLAS-like extension routines ###

# SDGMM, DDGMM, CDGMM, ZDGMM
_libcublas.cublasSdgmm.restype = int
_libcublas.cublasSdgmm.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc):
    """
    Matrix-diagonal matrix product for real general matrix.
        
    """

    status = _libcublas.cublasSdgmm(handle,
                                    _CUBLAS_SIDE[mode],
                                    m, n, 
                                    int(A), lda, 
                                    int(x), incx,
                                    int(C), ldc)
    cublasCheckStatus(status)
  
_libcublas.cublasDdgmm.restype = int
_libcublas.cublasDdgmm.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc):
    """
    Matrix-diagonal matrix product for real general matrix.
        
    """

    status = _libcublas.cublasDdgmm(handle,
                                    _CUBLAS_SIDE[mode],
                                    m, n, 
                                    int(A), lda, 
                                    int(x), incx,
                                    int(C), ldc)
    cublasCheckStatus(status)
    
_libcublas.cublasCdgmm.restype = int
_libcublas.cublasCdgmm.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasCdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """
    Matrix-diagonal matrix product for complex general matrix.
        
    """

    status = _libcublas.cublasCdgmm(handle,
                                    _CUBLAS_SIDE[mode],
                                    m, n, 
                                    int(A), lda, 
                                    int(x), incx,
                                    int(C), ldc)
    cublasCheckStatus(status)
      
_libcublas.cublasZdgmm.restype = int
_libcublas.cublasZdgmm.argtypes = [ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int]

def cublasZdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """
    Matrix-diagonal matrix product for complex general matrix.
        
    """

    status = _libcublas.cublasZdgmm(handle,
                                    _CUBLAS_SIDE[mode],
                                    m, n, 
                                    int(A), lda, 
                                    int(x), incx,
                                    int(C), ldc)
    cublasCheckStatus(status)        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
