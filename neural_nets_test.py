import unittest
import numpy as np
import pycuda.autoinit
from neural_nets.pycuda_ops.convolution import conv1d_matrix, \
     conv1d_matrix_mult_filter
from pycuda import gpuarray
from pycuda.curandom import rand as curand

FLOAT_ERR_TOL = 1e-3
DOUBLE_ERR_TOL = 1e-10

class TestConvolutionMatrix(unittest.TestCase):
    @staticmethod
    def cpu_conv1d(x, w):
        n, m = x.shape
        filter_width = w.size
        pad_width = filter_width // 2
        
        x_padded = np.concatenate((np.zeros((n, pad_width), dtype=np.float64),
                                   x,
                                   np.zeros((n, pad_width), dtype=np.float64)), 1)
        y = np.empty_like(x)
        for i in range(m):
            y[:,i] = (x_padded[:,i:(i+filter_width)]*w).sum(1)
        return y
    
    def conv1d_test_setup(self, height, width, filter_width):
        for dtype, err_tol in ((np.float32, FLOAT_ERR_TOL), 
                               (np.float64, DOUBLE_ERR_TOL)):
            x = curand((height, width), dtype)
            y = gpuarray.empty_like(x)
            w = curand((filter_width,), dtype)

            conv1d_matrix(x, w, y)
            y_cpu = self.cpu_conv1d(x.get(), w.get())

            self.assertLess(np.linalg.norm(y.get() - y_cpu, np.inf), err_tol)
    
    def test_conv1_matrix_small(self):
        for i in range(100):
            n = np.random.randint(4, 9)
            m = np.random.randint(4, 9)

            self.conv1d_test_setup(n, m, 5)

    def test_conv1_matrix_big_height(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(4, 9)
            self.conv1d_test_setup(n, m, 5)

    def test_conv1_matrix_big_width(self):
        for i in range(100):
            n = np.random.randint(4, 9)
            m = np.random.randint(200, 1000)
            self.conv1d_test_setup(n, m, 5)

    def test_conv1_matrix_big(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            self.conv1d_test_setup(n, m, 5)

    def test_conv1_matrix_big_filter(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            w = np.random.randint(5, 15)
            self.conv1d_test_setup(n, m, w)

class TestConvolutionMatrixMultFilters(unittest.TestCase):
    @staticmethod
    def cpu_conv1d(x, w, stride=1):
        n, m = x.shape        
        n_filters, filter_width = w.shape
        pad_width = filter_width // 2
        
        x_padded = np.concatenate((np.zeros((n, pad_width), dtype=np.float64),
                                   x,
                                   np.zeros((n, pad_width), dtype=np.float64)), 1)
        m_output = x.shape[1] // stride
        y = np.empty((n_filters, x.shape[0], m_output), dtype=x.dtype)
        # print x.shape, y.shape, stride
        # import pudb
        # pudb.set_trace()
        for f in range(n_filters):
            for i in range(0, m_output):
                ii = i*stride
                y[f,:,i] = (x_padded[:,ii:(ii+filter_width)]*w[f,:]).sum(1)
        return y


    @staticmethod    
    def gpu_conv1d(x, w, dtype, stride=1):
        y = conv1d_matrix_mult_filter(x, w, stride=stride)
        return y
    
    def conv1d_test_setup(self, height, width, filter_width, n_filters, stride=1):
        for dtype, err_tol in ((np.float32, FLOAT_ERR_TOL), 
                               (np.float64, DOUBLE_ERR_TOL)):
            x = curand((height, width), dtype)
            w = curand((n_filters, filter_width), dtype=dtype)
            y = self.gpu_conv1d(x, w, dtype, stride)
            y_np = self.cpu_conv1d(x.get(), w.get(), stride=stride)
            y_cpu = y.get()
            
            for f in range(n_filters):
                self.assertLess(np.linalg.norm(y_cpu[f,:,:]-y_np[f,:,:], np.inf), err_tol)
    
    def test_conv1_matrix_small(self):
        for i in range(100):
            n = np.random.randint(4, 9)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)

            self.conv1d_test_setup(n, m, 5, n_filters)

    def test_conv1_matrix_big_height(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, 5, n_filters)

    def test_conv1_matrix_big_width(self):
        for i in range(100):
            n = np.random.randint(4, 9)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, 5, n_filters)

    def test_conv1_matrix_big(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, 5, n_filters)

    def test_conv1_matrix_big_filter(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            w = np.random.randint(5, 15)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, w, n_filters)

    def test_conv1_matrix_stride(self):
        for i in range(100):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            filter_width = 5
            n_filters = np.random.randint(2, 5)
            stride = np.random.randint(2, 5)

            for dtype, err_tol in ((np.float32, FLOAT_ERR_TOL), 
                                   (np.float64, DOUBLE_ERR_TOL)):
                x = curand((n, m), dtype)
                w = curand((n_filters, filter_width), dtype=dtype)

                y_no_stride = self.gpu_conv1d(x, w, dtype, 1)
                y_stride = self.gpu_conv1d(x, w, dtype, stride)

                for f in range(n_filters):                
                    self.assertLess(np.linalg.norm(y_no_stride.get()[f,:,::stride]-
                                                   y_stride.get()[f,:,:],
                                    np.inf), err_tol)

if __name__ == '__main__':
    unittest.main()
