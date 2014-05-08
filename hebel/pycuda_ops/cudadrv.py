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
Python interface to CUDA driver functions.
"""

import sys, ctypes

# Load CUDA driver library:
if sys.platform == 'linux2':
    _libcuda_libname_list = ['libcuda.so', 'libcuda.so.3', 'libcuda.so.4']
elif sys.platform == 'darwin':
    _libcuda_libname_list = ['libcuda.dylib']
elif sys.platform == 'win32':
    _libcuda_libname_list = ['nvcuda.dll']
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcuda = None
for _libcuda_libname in _libcuda_libname_list:
    try:
        _libcuda = ctypes.cdll.LoadLibrary(_libcuda_libname)
    except OSError:
        pass
    else:
        break
if _libcuda == None:
    raise OSError('CUDA driver library not found')


# Exceptions corresponding to various CUDA driver errors:

class CUDA_ERROR(Exception):
    """CUDA error."""
    pass

class CUDA_ERROR_INVALID_VALUE(CUDA_ERROR):
    pass

class CUDA_ERROR_OUT_OF_MEMORY(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_INITIALIZED(CUDA_ERROR):
    pass

class CUDA_ERROR_DEINITIALIZED(CUDA_ERROR):
    pass

class CUDA_ERROR_PROFILER_DISABLED(CUDA_ERROR):
    pass

class CUDA_ERROR_PROFILER_NOT_INITIALIZED(CUDA_ERROR):
    pass

class CUDA_ERROR_PROFILER_ALREADY_STARTED(CUDA_ERROR):
    pass

class CUDA_ERROR_PROFILER_ALREADY_STOPPED(CUDA_ERROR):
    pass

class CUDA_ERROR_NO_DEVICE(CUDA_ERROR):
    pass

class CUDA_ERROR_INVALID_DEVICE(CUDA_ERROR):
    pass

class CUDA_ERROR_INVALID_IMAGE(CUDA_ERROR):
    pass

class CUDA_ERROR_INVALID_CONTEXT(CUDA_ERROR):
    pass

class CUDA_ERROR_CONTEXT_ALREADY_CURRENT(CUDA_ERROR):
    pass

class CUDA_ERROR_MAP_FAILED(CUDA_ERROR):
    pass

class CUDA_ERROR_UNMAP_FAILED(CUDA_ERROR):
    pass

class CUDA_ERROR_ARRAY_IS_MAPPED(CUDA_ERROR):
    pass

class CUDA_ERROR_ALREADY_MAPPED(CUDA_ERROR):
    pass

class CUDA_ERROR_NO_BINARY_FOR_GPU(CUDA_ERROR):
    pass

class CUDA_ERROR_ALREADY_ACQUIRED(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_MAPPED(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_MAPPED_AS_ARRAY(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_MAPPED_AS_POINTER(CUDA_ERROR):
    pass

class CUDA_ERROR_ECC_UNCORRECTABLE(CUDA_ERROR):
    pass

class CUDA_ERROR_UNSUPPORTED_LIMIT(CUDA_ERROR):
    pass

class CUDA_ERROR_CONTEXT_ALREADY_IN_USE(CUDA_ERROR):
    pass

class CUDA_ERROR_INVALID_SOURCE(CUDA_ERROR):
    pass

class CUDA_ERROR_FILE_NOT_FOUND(CUDA_ERROR):
    pass

class CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND(CUDA_ERROR):
    pass

class CUDA_ERROR_SHARED_OBJECT_INIT_FAILED(CUDA_ERROR):
    pass

class CUDA_ERROR_OPERATING_SYSTEM(CUDA_ERROR):
    pass

class CUDA_ERROR_INVALID_HANDLE(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_FOUND(CUDA_ERROR):
    pass

class CUDA_ERROR_NOT_READY(CUDA_ERROR):
    pass


CUDA_EXCEPTIONS = {
    1: CUDA_ERROR_INVALID_VALUE,
    2: CUDA_ERROR_OUT_OF_MEMORY,
    3: CUDA_ERROR_NOT_INITIALIZED,
    4: CUDA_ERROR_DEINITIALIZED,
    5: CUDA_ERROR_PROFILER_DISABLED,
    6: CUDA_ERROR_PROFILER_NOT_INITIALIZED,
    7: CUDA_ERROR_PROFILER_ALREADY_STARTED,
    8: CUDA_ERROR_PROFILER_ALREADY_STOPPED,
    100: CUDA_ERROR_NO_DEVICE,
    101: CUDA_ERROR_INVALID_DEVICE,
    200: CUDA_ERROR_INVALID_IMAGE,
    201: CUDA_ERROR_INVALID_CONTEXT,
    202: CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
    205: CUDA_ERROR_MAP_FAILED,
    206: CUDA_ERROR_UNMAP_FAILED,
    207: CUDA_ERROR_ARRAY_IS_MAPPED,
    208: CUDA_ERROR_ALREADY_MAPPED,
    209: CUDA_ERROR_NO_BINARY_FOR_GPU,
    210: CUDA_ERROR_ALREADY_ACQUIRED,
    211: CUDA_ERROR_NOT_MAPPED,
    212: CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
    213: CUDA_ERROR_NOT_MAPPED_AS_POINTER,
    214: CUDA_ERROR_ECC_UNCORRECTABLE,
    215: CUDA_ERROR_UNSUPPORTED_LIMIT,
    216: CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
    300: CUDA_ERROR_INVALID_SOURCE,
    301: CUDA_ERROR_FILE_NOT_FOUND,
    302: CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
    303: CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
    304: CUDA_ERROR_OPERATING_SYSTEM,
    400: CUDA_ERROR_INVALID_HANDLE,
    500: CUDA_ERROR_NOT_FOUND,
    600: CUDA_ERROR_NOT_READY,
    }

def cuCheckStatus(status):
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA driver
    error code.

    Parameters
    ----------
    status : int
        CUDA driver error code.

    See Also
    --------
    CUDA_EXCEPTIONS

    """

    if status != 0:
        try:
            raise CUDA_EXCEPTIONS[status]
        except KeyError:
            raise CUDA_ERROR

        
CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2 
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4

_libcuda.cuPointerGetAttribute.restype = int
_libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p,
                                           ctypes.c_int,
                                           ctypes.c_uint]
def cuPointerGetAttribute(attribute, ptr):
    data = ctypes.c_void_p()
    status = _libcuda.cuPointerGetAttribute(data, attribute, ptr)
    cuCheckStatus(status)
    return data
