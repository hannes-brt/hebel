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
Python interface to CUDA runtime functions.
"""

import atexit, ctypes, sys, warnings
import numpy as np

# Load CUDA runtime library:
if sys.platform == 'linux2':
    _libcudart_libname_list = ['libcudart.so', 'libcudart.so.3', 'libcudart.so.4']
elif sys.platform == 'darwin':
    _libcudart_libname_list = ['libcudart.dylib']
elif sys.platform == 'win32':
    _libcudart_libname_list = ['cudart64_60.dll', 'cudart32_60.dll', 
                               'cudart64_55.dll', 'cudart32_55.dll', 
                               'cudart64_50.dll', 'cudart32_50.dll']
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcudart = None
for _libcudart_libname in _libcudart_libname_list:
    try:
        _libcudart = ctypes.cdll.LoadLibrary(_libcudart_libname)
    except OSError:
        pass
    else:
        break
if _libcudart == None:
    raise OSError('CUDA runtime library not found')

# Code adapted from PARRET:
def POINTER(obj):
    """
    Create ctypes pointer to object.

    Notes
    -----
    This function converts None to a real NULL pointer because of bug
    in how ctypes handles None on 64-bit platforms.

    """

    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

# Classes corresponding to CUDA vector structures:
class float2(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float)
        ]

class cuFloatComplex(float2):
    @property
    def value(self):
        return complex(self.x, self.y)

class double2(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double)
        ]

class cuDoubleComplex(double2):
    @property
    def value(self):
        return complex(self.x, self.y)

def gpuarray_ptr(g):
    """
    Return ctypes pointer to data in GPUAarray object.

    """

    addr = int(g.gpudata)
    if g.dtype == np.int8:
        return ctypes.cast(addr, POINTER(ctypes.c_byte))
    if g.dtype == np.uint8:
        return ctypes.cast(addr, POINTER(ctypes.c_ubyte))
    if g.dtype == np.int16:
        return ctypes.cast(addr, POINTER(ctypes.c_short))
    if g.dtype == np.uint16:
        return ctypes.cast(addr, POINTER(ctypes.c_ushort))
    if g.dtype == np.int32:
        return ctypes.cast(addr, POINTER(ctypes.c_int))
    if g.dtype == np.uint32:
        return ctypes.cast(addr, POINTER(ctypes.c_uint))
    if g.dtype == np.int64:
        return ctypes.cast(addr, POINTER(ctypes.c_long))
    if g.dtype == np.uint64:
        return ctypes.cast(addr, POINTER(ctypes.c_ulong))
    if g.dtype == np.float32:
        return ctypes.cast(addr, POINTER(ctypes.c_float))
    elif g.dtype == np.float64:
        return ctypes.cast(addr, POINTER(ctypes.c_double))
    elif g.dtype == np.complex64:
        return ctypes.cast(addr, POINTER(cuFloatComplex))
    elif g.dtype == np.complex128:
        return ctypes.cast(addr, POINTER(cuDoubleComplex))
    else:
        raise ValueError('unrecognized type')

_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
def cudaGetErrorString(e):
    """
    Retrieve CUDA error string.

    Return the string associated with the specified CUDA error status
    code.

    Parameters
    ----------
    e : int
        Error number.

    Returns
    -------
    s : str
        Error string.

    """

    return _libcudart.cudaGetErrorString(e)

# Generic CUDA error:
class cudaError(Exception):
    """CUDA error."""
    pass

# Exceptions corresponding to various CUDA runtime errors:
class cudaErrorMissingConfiguration(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(1)
    pass

class cudaErrorMemoryAllocation(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(2)
    pass

class cudaErrorInitializationError(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(3)
    pass

class cudaErrorLaunchFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(4)
    pass

class cudaErrorPriorLaunchFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(5)
    pass

class cudaErrorLaunchTimeout(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(6)
    pass

class cudaErrorLaunchOutOfResources(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(7)
    pass

class cudaErrorInvalidDeviceFunction(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(8)
    pass

class cudaErrorInvalidConfiguration(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(9)
    pass

class cudaErrorInvalidDevice(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(10)
    pass

class cudaErrorInvalidValue(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(11)
    pass

class cudaErrorInvalidPitchValue(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(12)
    pass

class cudaErrorInvalidSymbol(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(13)
    pass

class cudaErrorMapBufferObjectFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(14)
    pass

class cudaErrorUnmapBufferObjectFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(15)
    pass

class cudaErrorInvalidHostPointer(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(16)
    pass

class cudaErrorInvalidDevicePointer(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(17)
    pass

class cudaErrorInvalidTexture(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(18)
    pass

class cudaErrorInvalidTextureBinding(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(19)
    pass

class cudaErrorInvalidChannelDescriptor(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(20)
    pass

class cudaErrorInvalidMemcpyDirection(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(21)
    pass

class cudaErrorTextureFetchFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(23)
    pass

class cudaErrorTextureNotBound(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(24)
    pass

class cudaErrorSynchronizationError(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(25)
    pass

class cudaErrorInvalidFilterSetting(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(26)
    pass

class cudaErrorInvalidNormSetting(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(27)
    pass

class cudaErrorMixedDeviceExecution(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(28)
    pass

class cudaErrorCudartUnloading(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(29)
    pass

class cudaErrorUnknown(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(30)
    pass

class cudaErrorNotYetImplemented(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(31)
    pass

class cudaErrorMemoryValueTooLarge(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(32)
    pass

class cudaErrorInvalidResourceHandle(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(33)
    pass

class cudaErrorNotReady(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(34)
    pass

class cudaErrorInsufficientDriver(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(35)
    pass

class cudaErrorSetOnActiveProcess(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(36)
    pass

class cudaErrorInvalidSurface(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(37)
    pass

class cudaErrorNoDevice(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(38)
    pass

class cudaErrorECCUncorrectable(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(39)
    pass

class cudaErrorSharedObjectSymbolNotFound(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(40)
    pass

class cudaErrorSharedObjectInitFailed(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(41)
    pass

class cudaErrorUnsupportedLimit(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(42)
    pass

class cudaErrorDuplicateVariableName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(43)
    pass

class cudaErrorDuplicateTextureName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(44)
    pass

class cudaErrorDuplicateSurfaceName(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(45)
    pass

class cudaErrorDevicesUnavailable(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(46)
    pass

class cudaErrorInvalidKernelImage(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(47)
    pass

class cudaErrorNoKernelImageForDevice(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(48)
    pass

class cudaErrorIncompatibleDriverContext(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(49)
    pass

class cudaErrorPeerAccessAlreadyEnabled(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(50)
    pass

class cudaErrorPeerAccessNotEnabled(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(51)
    pass

class cudaErrorDeviceAlreadyInUse(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(54)
    pass

class cudaErrorProfilerDisabled(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(55)
    pass

class cudaErrorProfilerNotInitialized(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(56)
    pass

class cudaErrorProfilerAlreadyStarted(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(57)
    pass

class cudaErrorProfilerAlreadyStopped(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(58)
    pass

class cudaErrorStartupFailure(cudaError):
    __doc__ = _libcudart.cudaGetErrorString(127)
    pass

cudaExceptions = {
    1: cudaErrorMissingConfiguration,
    2: cudaErrorMemoryAllocation,
    3: cudaErrorInitializationError,
    4: cudaErrorLaunchFailure,
    5: cudaErrorPriorLaunchFailure,
    6: cudaErrorLaunchTimeout,
    7: cudaErrorLaunchOutOfResources,
    8: cudaErrorInvalidDeviceFunction,
    9: cudaErrorInvalidConfiguration,
    10: cudaErrorInvalidDevice,
    11: cudaErrorInvalidValue,
    12: cudaErrorInvalidPitchValue,
    13: cudaErrorInvalidSymbol,
    14: cudaErrorMapBufferObjectFailed,
    15: cudaErrorUnmapBufferObjectFailed,
    16: cudaErrorInvalidHostPointer,
    17: cudaErrorInvalidDevicePointer,
    18: cudaErrorInvalidTexture,
    19: cudaErrorInvalidTextureBinding,
    20: cudaErrorInvalidChannelDescriptor,
    21: cudaErrorInvalidMemcpyDirection,
    22: cudaError,
    23: cudaErrorTextureFetchFailed,
    24: cudaErrorTextureNotBound,
    25: cudaErrorSynchronizationError,
    26: cudaErrorInvalidFilterSetting,
    27: cudaErrorInvalidNormSetting,
    28: cudaErrorMixedDeviceExecution,
    29: cudaErrorCudartUnloading,
    30: cudaErrorUnknown,
    31: cudaErrorNotYetImplemented,
    32: cudaErrorMemoryValueTooLarge,
    33: cudaErrorInvalidResourceHandle,
    34: cudaErrorNotReady,
    35: cudaErrorInsufficientDriver,
    36: cudaErrorSetOnActiveProcess,
    37: cudaErrorInvalidSurface,
    38: cudaErrorNoDevice,
    39: cudaErrorECCUncorrectable,
    40: cudaErrorSharedObjectSymbolNotFound,
    41: cudaErrorSharedObjectInitFailed,
    42: cudaErrorUnsupportedLimit,
    43: cudaErrorDuplicateVariableName,
    44: cudaErrorDuplicateTextureName,
    45: cudaErrorDuplicateSurfaceName,
    46: cudaErrorDevicesUnavailable,
    47: cudaErrorInvalidKernelImage,
    48: cudaErrorNoKernelImageForDevice,
    49: cudaErrorIncompatibleDriverContext,
    50: cudaErrorPeerAccessAlreadyEnabled,
    51: cudaErrorPeerAccessNotEnabled,
    52: cudaError,
    53: cudaError,
    54: cudaErrorDeviceAlreadyInUse,
    55: cudaErrorProfilerDisabled,
    56: cudaErrorProfilerNotInitialized,
    57: cudaErrorProfilerAlreadyStarted,
    58: cudaErrorProfilerAlreadyStopped,
    127: cudaErrorStartupFailure
    }

def cudaCheckStatus(status):
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA runtime error
    code.

    Parameters
    ----------
    status : int
        CUDA runtime error code.

    See Also
    --------
    cudaExceptions

    """

    if status != 0:
        try:
            raise cudaExceptions[status]
        except KeyError:
            raise cudaError

# Memory allocation functions (adapted from pystream):
_libcudart.cudaMalloc.restype = int
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_size_t]
def cudaMalloc(count, ctype=None):
    """
    Allocate device memory.

    Allocate memory on the device associated with the current active
    context.

    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), count)
    cudaCheckStatus(status)
    if ctype != None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr

_libcudart.cudaFree.restype = int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
def cudaFree(ptr):
    """
    Free device memory.

    Free allocated memory on the device associated with the current active
    context.

    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    status = _libcudart.cudaFree(ptr)
    cudaCheckStatus(status)

_libcudart.cudaMallocPitch.restype = int
_libcudart.cudaMallocPitch.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.c_size_t, ctypes.c_size_t]
def cudaMallocPitch(pitch, rows, cols, elesize):
    """
    Allocate pitched device memory.

    Allocate pitched memory on the device associated with the current active
    context.

    Parameters
    ----------
    pitch : int
        Pitch for allocation.
    rows : int
        Requested pitched allocation height.
    cols : int
        Requested pitched allocation width.
    elesize : int
        Size of memory element.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    ptr = ctypes.c_void_p()
    status = _libcudart.cudaMallocPitch(ctypes.byref(ptr),
                                        ctypes.c_size_t(pitch), cols*elesize,
                                        rows)
    cudaCheckStatus(status)
    return ptr, pitch

# Memory copy modes:
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

_libcudart.cudaMemcpy.restype = int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
def cudaMemcpy_htod(dst, src, count):
    """
    Copy memory from host to device.

    Copy data from host memory to device memory.

    Parameters
    ----------
    dst : ctypes pointer
        Device memory pointer.
    src : ctypes pointer
        Host memory pointer.
    count : int
        Number of bytes to copy.

    """

    status = _libcudart.cudaMemcpy(dst, src,
                                   ctypes.c_size_t(count),
                                   cudaMemcpyHostToDevice)
    cudaCheckStatus(status)

def cudaMemcpy_dtoh(dst, src, count):
    """
    Copy memory from device to host.

    Copy data from device memory to host memory.

    Parameters
    ----------
    dst : ctypes pointer
        Host memory pointer.
    src : ctypes pointer
        Device memory pointer.
    count : int
        Number of bytes to copy.

    """

    status = _libcudart.cudaMemcpy(dst, src,
                                   ctypes.c_size_t(count),
                                   cudaMemcpyDeviceToHost)
    cudaCheckStatus(status)

_libcudart.cudaMemGetInfo.restype = int
_libcudart.cudaMemGetInfo.argtypes = [ctypes.c_void_p,
                                      ctypes.c_void_p]
def cudaMemGetInfo():
    """
    Return the amount of free and total device memory.

    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.

    """

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libcudart.cudaMemGetInfo(ctypes.byref(free),
                                       ctypes.byref(total))
    cudaCheckStatus(status)
    return free.value, total.value

_libcudart.cudaSetDevice.restype = int
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
def cudaSetDevice(dev):
    """
    Set current CUDA device.

    Select a device to use for subsequent CUDA operations.

    Parameters
    ----------
    dev : int
        Device number.

    """

    status = _libcudart.cudaSetDevice(dev)
    cudaCheckStatus(status)

_libcudart.cudaGetDevice.restype = int
_libcudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
def cudaGetDevice():
    """
    Get current CUDA device.

    Return the identifying number of the device currently used to
    process CUDA operations.

    Returns
    -------
    dev : int
        Device number.

    """

    dev = ctypes.c_int()
    status = _libcudart.cudaGetDevice(ctypes.byref(dev))
    cudaCheckStatus(status)
    return dev.value

_libcudart.cudaDriverGetVersion.restype = int
_libcudart.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
def cudaDriverGetVersion():
    """
    Get installed CUDA driver version.

    Return the version of the installed CUDA driver as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Driver version.

    """

    version = ctypes.c_int()
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    cudaCheckStatus(status)
    return version.value

# Memory types:
cudaMemoryTypeHost = 1
cudaMemoryTypeDevice = 2

class cudaPointerAttributes(ctypes.Structure):
    _fields_ = [
        ('memoryType', ctypes.c_int),
        ('device', ctypes.c_int),
        ('devicePointer', ctypes.c_void_p),
        ('hostPointer', ctypes.c_void_p)
        ]

_libcudart.cudaPointerGetAttributes.restype = int
_libcudart.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p,
                                                ctypes.c_void_p]
def cudaPointerGetAttributes(ptr):
    """
    Get memory pointer attributes.

    Returns attributes of the specified pointer.

    Parameters
    ----------
    ptr : ctypes pointer
        Memory pointer to examine.

    Returns
    -------
    memory_type : int
        Memory type; 1 indicates host memory, 2 indicates device
        memory.
    device : int
        Number of device associated with pointer.

    Notes
    -----
    This function only works with CUDA 4.0 and later.

    """

    attributes = cudaPointerAttributes()
    status = \
        _libcudart.cudaPointerGetAttributes(ctypes.byref(attributes), ptr)
    cudaCheckStatus(status)
    return attributes.memoryType, attributes.device

