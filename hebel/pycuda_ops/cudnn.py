import libcudnn, ctypes, atexit
from .. import memory_pool, context
from pycuda import gpuarray
import numpy as np

_global_cudnn_handle = None
def init():
    global _global_cudnn_handle
    _global_cudnn_handle = libcudnn.cudnnCreate()

@atexit.register
def destroy_handle():
    global _global_cudnn_handle
    if _global_cudnn_handle is not None:
        libcudnn.cudnnDestroy(_global_cudnn_handle)

class Tensor4dDesc(object):
    def __init__(self, n, c, h, w, double=False):
        self.double = double
        if double:
            self.data_type = libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']
        else:
            self.data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
        self.handle = libcudnn.cudnnCreateTensor4dDescriptor()
        libcudnn.cudnnSetTensor4dDescriptor(
            self.handle, libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            self.data_type, n, c, h, w
        )
        self.n = n
        self.c = c
        self.h = h
        self.w = w

    @classmethod
    def from_gpuarray(cls, x):
        n, c, h, w = x.shape
        if x.dtype == np.float32:
            double = False
        elif x.dtype == np.float64:
            double = True
        else:
            raise ValueError("incompatible data type")

        return cls(n, c, h, w, double)
        
    def __del__(self):
        libcudnn.cudnnDestroyTensor4dDescriptor(self.handle)

class FilterDesc(object):
    def __init__(self, k, c, h, w, double=False):
        self.double = double
        if double:
            self.data_type = libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']
        else:
            self.data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
        self.handle = libcudnn.cudnnCreateFilterDescriptor()
        libcudnn.cudnnSetFilterDescriptor(self.handle, self.data_type, k, c, h, w)
        self.k = k
        self.c = c
        self.h = h
        self.w = w

    @classmethod
    def from_gpuarray(cls, x):
        k, c, h, w = x.shape
        if x.dtype == np.float32:
            double = False
        elif x.dtype == np.float64:
            double = True
        else:
            raise ValueError("incompatible data type")

        return cls(k, c, h, w, double)

    def __del__(self):
        libcudnn.cudnnDestroyFilterDescriptor(self.handle)

class ConvolutionDesc(object):
    def __init__(self, bottom_desc, filter_desc, pad_h=0, pad_w=0, u=1, v=1):
        self.bottom_desc = bottom_desc
        self.filter_desc = filter_desc
        mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
        upscalex = 1
        upscaley = 1
        self.handle = libcudnn.cudnnCreateConvolutionDescriptor()
        libcudnn.cudnnSetConvolutionDescriptor(self.handle, bottom_desc.handle,
                                               filter_desc.handle, pad_h, pad_w, u, v,
                                               upscalex, upscaley, mode)
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.u = u
        self.v = v
        self._get_top_desc()
        self._get_grad_weights_tensor_desc()
        self._get_grad_data_tensor_desc()
        self._get_bias_tensor_desc()

    def _get_top_desc(self):
        n, c, h, w = libcudnn.cudnnGetOutputTensor4dDim(
            self.handle, libcudnn.cudnnConvolutionPath['CUDNN_CONVOLUTION_FORWARD'])
        self.top_desc = Tensor4dDesc(n, c, h, w, self.bottom_desc.double)

    def _get_grad_weights_tensor_desc(self):
        n, c, h, w = libcudnn.cudnnGetOutputTensor4dDim(
            self.handle, libcudnn.cudnnConvolutionPath['CUDNN_CONVOLUTION_WEIGHT_GRAD'])
        self.grad_weights_tensor_desc = Tensor4dDesc(n, c, h, w, self.bottom_desc.double)

    def _get_grad_data_tensor_desc(self):
        n, c, h, w = libcudnn.cudnnGetOutputTensor4dDim(
            self.handle, libcudnn.cudnnConvolutionPath['CUDNN_CONVOLUTION_DATA_PATH'])
        self.grad_weights_tensor_desc = Tensor4dDesc(n, c, h, w, self.bottom_desc.double)

    def _get_bias_tensor_desc(self):
        self.grad_bias_desc = Tensor4dDesc(1, self.top_desc.c, 1, 1, self.bottom_desc.double)

    def __del__(self):
        libcudnn.cudnnDestroyConvolutionDescriptor(self.handle)

class PoolingDesc(object):
    def __init__(self, window_height, window_width,
                 vertical_stride=1, horizontal_stride=1, mode='max'):
        if mode == 'max':
            self.mode = libcudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        elif mode == 'avg':
            self.mode = libcudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE']
        else:
            raise ValueError("unknown pooling mode")

        self.window_height = window_height
        self.window_width = window_width
        self.vertical_stride = vertical_stride
        self.horizontal_stride = horizontal_stride

        self.handle = libcudnn.cudnnCreatePoolingDescriptor()
        libcudnn.cudnnSetPoolingDescriptor(self.handle, self.mode, self.window_height,
                                           self.window_width, self.vertical_stride,
                                           self.horizontal_stride)

    def __del__(self):
        libcudnn.cudnnDestroyPoolingDescriptor(self.handle)

def convolution_forward(bottom, filter, bias,
                        conv_desc, top=None,
                        accumulate=False):
    if accumulate:
        mode = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_ACCUMULATE']
    else:
        mode = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_NO_ACCUMULATE']

    if top is None:
        d = conv_desc.top_desc
        top = gpuarray.empty(
            (d.n, d.c, d.h, d.w), (np.float32, np.float64)[d.data_type],
            allocator=memory_pool.allocate
        )

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    filter_data = ctypes.c_void_p(int(filter.gpudata))
    top_data = ctypes.c_void_p(int(top.gpudata))
    bias_data = ctypes.c_void_p(int(bias.gpudata))

    libcudnn.cudnnConvolutionForward(
        _global_cudnn_handle, conv_desc.bottom_desc.handle, bottom_data,
        conv_desc.filter_desc.handle, filter_data, conv_desc.handle,
        conv_desc.top_desc.handle, top_data, mode
    )

    alpha = [ctypes.c_float, ctypes.c_double][conv_desc.top_desc.data_type](1.)
    libcudnn.cudnnAddTensor4d(
        _global_cudnn_handle, libcudnn.cudnnAddMode['CUDNN_ADD_SAME_C'],
        ctypes.byref(alpha), conv_desc.grad_bias_desc.handle,
        bias_data, conv_desc.top_desc.handle, top_data
    )

    return top

def convolution_backward(bottom, filter, top_diff, conv_desc,
                         grad_bias=None, grad_filter=None,
                         grad_input=None, accumulate=False):
    if accumulate:
        accumulate = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_ACCUMULATE']
    else:
        accumulate = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_NO_ACCUMULATE']

    if grad_bias is None:
        d = conv_desc.grad_bias_desc
        grad_bias = gpuarray.empty(
            (1, d.c, 1, 1), (np.float32, np.float64)[d.data_type],
            allocator=memory_pool.allocate
        )

    if grad_filter is None:
        d = conv_desc.filter_desc
        grad_filter = gpuarray.empty(
            (d.k, d.c, d.h, d.w), (np.float32, np.float64)[d.data_type],
            allocator=memory_pool.allocate
        )

    if grad_input is None:
        d = conv_desc.bottom_desc
        grad_input = gpuarray.empty(
            (d.n, d.c, d.h, d.w), (np.float32, np.float64)[d.data_type],
            allocator=memory_pool.allocate
        )

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    filter_data = ctypes.c_void_p(int(filter.gpudata))
    top_diff_data = ctypes.c_void_p(int(top_diff.gpudata))
    grad_bias_data = ctypes.c_void_p(int(grad_bias.gpudata))
    grad_filter_data = ctypes.c_void_p(int(grad_filter.gpudata))
    grad_input_data = ctypes.c_void_p(int(grad_input.gpudata))

    libcudnn.cudnnConvolutionBackwardBias(
        _global_cudnn_handle, conv_desc.top_desc.handle, top_diff_data,
        conv_desc.grad_bias_desc.handle, grad_bias_data, accumulate
    )

    libcudnn.cudnnConvolutionBackwardFilter(
        _global_cudnn_handle, conv_desc.bottom_desc.handle, bottom_data,
        conv_desc.top_desc.handle, top_diff_data,
        conv_desc.handle, conv_desc.filter_desc.handle, grad_filter_data, accumulate
    )

    libcudnn.cudnnConvolutionBackwardData(
        _global_cudnn_handle, conv_desc.filter_desc.handle, filter_data,
        conv_desc.top_desc.handle, top_diff_data, conv_desc.handle,
        conv_desc.bottom_desc.handle, grad_input_data, accumulate
    )

    return grad_bias, grad_filter, grad_input

def pooling_forward(pooling_desc, bottom,
                    top_desc=None, top=None,
                    bottom_desc=None):

    if top is None:
        n, c = bottom.shape[:2]
        h_out =  bottom.shape[2] // \
                 pooling_desc.vertical_stride // \
                 pooling_desc.window_height
        w_out = bottom.shape[3] // \
                pooling_desc.horizontal_stride // \
                pooling_desc.window_width
        top = gpuarray.empty((n, c, h_out, w_out), bottom.dtype,
                                   allocator=memory_pool.allocate)

    if top_desc is None:
        top_desc = Tensor4dDesc.from_gpuarray(top)

    if bottom_desc is None:
        bottom_desc = Tensor4dDesc.from_gpuarray(bottom)

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    top_data = ctypes.c_void_p(int(top.gpudata))

    libcudnn.cudnnPoolingForward(_global_cudnn_handle,
                                 pooling_desc.handle, bottom_desc.handle,
                                 bottom_data, top_desc.handle,
                                 top_data)

    return top

def pooling_backward(pooling_desc, bottom,
                     top, top_diff, bottom_desc=None,
                     top_desc=None, bottom_diff=None):

    if bottom_diff is None:
        bottom_diff = gpuarray.empty_like(bottom)

    if bottom_desc is None:
        bottom_desc = Tensor4dDesc.from_gpuarray(bottom)

    if top_desc is None:
        top_desc = Tensor4dDesc.from_gpuarray(top)

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    bottom_diff_data = ctypes.c_void_p(int(bottom_diff.gpudata))
    top_data = ctypes.c_void_p(int(top.gpudata))
    top_diff_data = ctypes.c_void_p(int(top_diff.gpudata))

    libcudnn.cudnnPoolingBackward(_global_cudnn_handle, pooling_desc.handle,
                                  top_desc.handle, top_data,
                                  top_desc.handle, top_diff_data,
                                  bottom_desc.handle, bottom_data,
                                  bottom_desc.handle, bottom_diff_data)
    return bottom_diff

def activation_forward(bottom, mode='relu', bottom_desc=None,
                       top=None, top_desc=None):

    if bottom_desc is None:
        bottom_desc = Tensor4dDesc.from_gpuarray(bottom)

    if top is None:
        top = gpuarray.empty_like(bottom)

    if top_desc is None:
        top_desc = Tensor4dDesc.from_gpuarray(top)

    if mode == 'relu':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']
    elif mode == 'tanh':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
    elif mode == 'sigmoid':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']
    else:
        raise ValueError("unknown activation function")

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    top_data = ctypes.c_void_p(int(top.gpudata))

    libcudnn.cudnnActivationForward(_global_cudnn_handle, mode,
                                    bottom_desc.handle, bottom_data,
                                    top_desc.handle, top_data)

    return top

def activation_backward(bottom, top, top_diff, mode='relu',
                        bottom_diff=None, bottom_desc=None,
                        top_desc=None):

    if bottom_diff is None:
        bottom_diff = gpuarray.empty_like(bottom)

    if bottom_desc is None:
        bottom_desc = Tensor4dDesc.from_gpuarray(bottom)

    if top_desc is None:
        top_desc = Tensor4dDesc.from_gpuarray(top)

    if mode == 'relu':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']
    elif mode == 'tanh':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
    elif mode == 'sigmoid':
        mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']
    else:
        raise ValueError("unknown activation function")

    bottom_data = ctypes.c_void_p(int(bottom.gpudata))
    bottom_diff_data = ctypes.c_void_p(int(bottom_diff.gpudata))
    top_data = ctypes.c_void_p(int(top.gpudata))
    top_diff_data = ctypes.c_void_p(int(top_diff.gpudata))

    libcudnn.cudnnActivationBackward(_global_cudnn_handle, mode,
                                     top_desc.handle, top_data,
                                     top_desc.handle, top_diff_data,
                                     bottom_desc.handle, bottom_data,
                                     bottom_desc.handle, bottom_diff_data)

    return bottom_diff