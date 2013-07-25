import unittest, random
import numpy as np
import pycuda.autoinit
from sequence_convolution.pycuda_ops import convolve_sequence, \
     convolve_sequence_gradient, max_pool, max_pool_gradient
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from sequence_convolution.sequence_convolution import SequenceConvolutionNet, \
     SequenceConvolutionLayer
from neural_nets.data_providers import MiniBatchDataProvider
from neural_nets.optimizers import SGD
from neural_nets.schedulers import constant_scheduler
from neural_nets.parameter_updaters import SimpleSGDUpdate
from neural_nets.monitors import SimpleProgressMonitor
from create_features import encode_seq
from copy import copy

import numpy as np

STRIDE = 4

def checkgrad_model(layer, input, epsilon=1e-4, **kwargs):
    cache = layer.feed_forward(input)
    f0 = np.sum(cache[0].get())
    df_output = gpuarray.empty_like(cache[0]).fill(1.)
    grads = layer.backprop(input, df_output, cache)[0]

    grad_approx = [np.empty_like(p.get()) for p in layer.parameters]
    loss = 0

    parameters = layer.parameters
    for i in range(len(layer.parameters)):        
        param_i = parameters[i].get()
        grad_approx_i = grad_approx[i]

        assert param_i.shape == grad_approx_i.shape

        for idx, _ in np.ndenumerate(grad_approx_i):
            p = list(copy(parameters))
            w0 = param_i[idx]

            # Get f(x - epsilon)
            param_i[idx] += epsilon
            p[i] = gpuarray.to_gpu(param_i)
            layer.parameters = p
            f1 = np.sum(layer.feed_forward(input)[0].get())

            # Get f(x + epsilon)
            param_i[idx] -= 2 * epsilon
            p[i] = gpuarray.to_gpu(param_i)
            layer.parameters = p
            f2 = np.sum(layer.feed_forward(input)[0].get())

            # Reset weight
            param_i[idx] = w0
            p[i] = gpuarray.to_gpu(param_i)
            layer.parameters = p

            # Compute gradient approximation
            grad_approx_i[idx] = (f1 - f2) / (2 * epsilon)

        loss += np.sum(((grads[i].get() - grad_approx_i) / grads[i].get()) ** 2.)
    loss = np.sqrt(loss)

    return loss

class TestConvolution(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-4
    DOUBLE_ERR_TOL = 1e-13
    
    @staticmethod
    def cpu_conv1d(x, w, b, stride=1):
        n, m = x.shape        
        n_filters, filter_width = w.shape

        pad_width = filter_width - 1
        x_padded = np.concatenate((x,
                                   np.zeros((n, pad_width), dtype=x.dtype)), 1)

        m_output = int(np.ceil(x.shape[1] / float(stride)))
        y = np.empty((x.shape[0], n_filters, m_output), dtype=x.dtype)
        for f in range(n_filters):
            for i in range(0, m_output):
                ii = i*stride
                y[:,f,i] = b[f] + (x_padded[:,ii:(ii+filter_width)]*w[f,:]).sum(1)
        return y

    @staticmethod    
    def gpu_conv1d(x, w, b, dtype, stride=1):
        y = convolve_sequence(x, w, b, stride=stride)
        return y
    
    def conv1d_test_setup(self, height, width, filter_width, n_filters, 
                          stride=STRIDE):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL), 
                               (np.float64, self.DOUBLE_ERR_TOL)):
            x = curand((height, width), dtype)
            w = curand((n_filters, filter_width), dtype=dtype)
            b = curand((n_filters,), dtype=dtype)
            y = self.gpu_conv1d(x, w, b, dtype, stride)
            y_np = self.cpu_conv1d(x.get(), w.get(), b.get(),
                                   stride=stride)
            y_cpu = y.get()
            
            # for f in range(n_filters):
            #     err = (y_cpu[f,:,:]-y_np[f,:,:])/y_cpu[f,:,:]
            #     self.assertLess(np.linalg.norm(err, np.inf), err_tol)
            self.assertLess(np.max((y_cpu - y_np) / y_cpu), err_tol)
    
    def test_conv1d_matrix_small(self):
        for i in range(20):
            n = np.random.randint(4, 9)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)

            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big_height(self):
        for i in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big_width(self):
        for i in range(20):
            n = np.random.randint(4, 9)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big(self):
        for i in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big_filter(self):
        for i in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            w = 2*np.random.randint(2, 5)
            n_filters = np.random.randint(2, 5)            
            self.conv1d_test_setup(n, m, w, n_filters)

    def test_conv1d_matrix_stride(self):
        for i in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            filter_width = 4
            n_filters = np.random.randint(2, 5)
            stride = np.random.randint(2, 5)

            for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL), 
                                   (np.float64, self.DOUBLE_ERR_TOL)):
                x = curand((n, m), dtype)
                w = curand((n_filters, filter_width), dtype=dtype)
                b = curand((n_filters,), dtype=dtype)

                y_no_stride = self.gpu_conv1d(x, w, b, dtype, 1)
                y_stride = self.gpu_conv1d(x, w, b, dtype, stride)

                for f in range(n_filters):     
                    err = (y_no_stride.get()[f,:,::stride]-
                           y_stride.get()[f,:,:])/y_no_stride.get()[f,:,::stride]
                    self.assertLess(np.linalg.norm(err, np.inf), err_tol)

class TestConvolutionGradWeights(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-3
    DOUBLE_ERR_TOL = 1e-12
    
    @staticmethod
    def grad_weights_cpu(input, df_output, n_filters, filter_width):
        df_w = np.empty((n_filters, filter_width))
        for n in range(n_filters):
            for i in range(filter_width):
                input_padded = np.concatenate((input[:,i::], 
                                               np.zeros((input.shape[0], i))), 1)
                df_w[n, i] = (input_padded[:,::STRIDE]*df_output[:,n]).sum()
        return df_w

    def grad_weights_test(self, height, width, n_filters, filter_width):
        for dtype, err_tol in ((np.float64, self.DOUBLE_ERR_TOL),
                               (np.float32, self.FLOAT_ERR_TOL)):

            eps = np.finfo(dtype).eps
            x = curand((height, width), dtype)
            df_output = curand((height, n_filters, width // STRIDE), dtype)
            # df_output = gpuarray.empty((n_filters, height, width // STRIDE), dtype).fill(1.)

            df_w = convolve_sequence_gradient(x, df_output, filter_width, n_filters, block_size=1024)
            df_w_cpu = df_w.get()
            df_w_np = self.grad_weights_cpu(x.get(), df_output.get(), n_filters, filter_width)

            del df_w, x, df_output
            self.assertLess(np.abs((df_w_cpu-df_w_np)/(df_w_cpu+eps)).max(), err_tol)

    def test_grad_weights_filter_12(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = 12
            m = STRIDE*np.random.randint(filter_width/STRIDE, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_16(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = 16
            m = STRIDE*np.random.randint(filter_width/STRIDE, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_24(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = 24
            m = STRIDE*np.random.randint(filter_width/STRIDE, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_8(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = 8
            m = STRIDE*np.random.randint(filter_width/STRIDE, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_4(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = 4
            m = STRIDE*np.random.randint(filter_width/STRIDE, 200)
            n_filters = np.random.randint(2, 100)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights(self):
        for n in range(20):
            n = np.random.randint(5, 300)
            filter_width = STRIDE*np.random.randint(2, 8)
            m = STRIDE*np.random.randint(filter_width/STRIDE, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_1_block(self):
        for n in range(20):
            n = 10
            m = 16
            filter_width = 8
            n_filters = 5
            self.grad_weights_test(n, m, n_filters, filter_width)


class TestMaxPool(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_cpu(mat, pool_size):
        output = np.empty((mat.shape[0], mat.shape[1], mat.shape[2] / pool_size))
        for n in range(output.shape[0]):
            for i in range(output.shape[2]):
                output[n,:,i] = np.max(mat[n,:, pool_size*i:pool_size*(i+1)], 1)

        return output

    def max_pool_test(self, height, width, n_filters, pool_size):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                               (np.float64, self.DOUBLE_ERR_TOL)):
            mat = curand((n_filters, height, width), dtype)
            target, argmax = max_pool(mat, pool_size)
            target_cpu = target.get()
            target_np = self.max_pool_cpu(mat.get(), pool_size)
            for n in range(n_filters):
                self.assertLess(np.linalg.norm(
                    (target_cpu[n] - target_np[n]) / target_cpu[n], np.inf), 
                    err_tol)

    def test_max_pool(self):
        for i in range(20):
            height = np.random.randint(100, 1000)
            width = np.random.randint(20, 500)
            n_filters = np.random.randint(2, 10)
            pool_size = np.random.randint(2, 15)
            self.max_pool_test(height, width, n_filters, pool_size)

class TestMaxPoolGradient(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_grad_cpu(mat, mat_pooled, argmax, df_output, pool_size):
        n_filters = mat.shape[1]
        height = mat.shape[0]
        n_pooled = mat_pooled.shape[2]        
        
        df_input = np.zeros_like(mat)
        for n in range(n_filters):
            for i in range(n_pooled):
                df_input[range(height), height*[n], argmax[:, n, i]] = df_output[:, n, i]
        
        return df_input

    def max_pool_grad_test(self, height, width, n_filters, pool_size):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                               (np.float64, self.DOUBLE_ERR_TOL)):
            mat = curand((n_filters, height, width), dtype)
            mat_pooled, argmax = max_pool(mat, pool_size)
            df_output = curand(mat_pooled.shape, dtype)
            df_input = max_pool_gradient(mat, argmax, df_output, pool_size)
            df_input_cpu = df_input.get()
            df_input_np = self.max_pool_grad_cpu(mat.get(), mat_pooled.get(), 
                                                 argmax.get(),
                                                 df_output.get(), pool_size)

            self.assertTrue(np.all(df_input_cpu == df_input_np))

    def test_max_pool_grad(self):
        for i in range(20):
            n = np.random.randint(100, 1000)
            m = np.random.randint(15, 500)
            n_filters = np.random.randint(2, 10)
            pool_size = np.random.randint(2, 15)
            self.max_pool_grad_test(n, m, n_filters, pool_size)

class TestConvNet(unittest.TestCase):
    def test_conv_net(self):
        s = ['A' + ''.join([random.choice('ACGT') for x in range(7)]) for x in range(100)] + \
        ['T' + ''.join([random.choice('ACGT') for x in range(7)]) for x in range(100)]
        seq = np.array(map(encode_seq, s), np.float32)
        targets = np.array(100*[[1., 0.]] + 100*[[0., 1.]], dtype=np.float32)

        shuffle_idx = np.random.permutation(len(s))
        seq = gpuarray.to_gpu(seq[shuffle_idx])
        targets = gpuarray.to_gpu(targets[shuffle_idx])

        test_error = 1
        for i in range(10):
            model = SequenceConvolutionNet(seq.shape[1], 2, 32, 5, 8, [], 
                                           activation_function='tanh')
            
            train_data = MiniBatchDataProvider(seq, targets, 10)

            optimizer = SGD(model, SimpleSGDUpdate, train_data, train_data,
                            learning_rate_schedule=constant_scheduler(1.),
                            progress_monitor=SimpleProgressMonitor())

            optimizer.run(20)
            test_error = np.min([optimizer.best_validation_loss, test_error])

        self.assertEqual(test_error, 0.)

class TestConvolutionGradient(unittest.TestCase):
    EPSILON = 1e-3
    TOL = 1e-4
    
    def test_convolution_gradient(self):
        for i in range(20):
            n_in = 36
            filter_width = 12
            n_filters = 4
            conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters, 
                                                  dtype=np.float64)
            x = curand((100, n_in), dtype=np.float64)
            loss = checkgrad_model(conv_layer, x, epsilon=self.EPSILON)
            self.assertLess(loss, self.TOL)
            
if __name__ == '__main__':
    unittest.main()
