import unittest, random
import numpy as np
import pycuda.autoinit
from sequence_convolution import DNA_A, DNA_C, DNA_G, DNA_T
from sequence_convolution.pycuda_ops import convolve_sequence, \
     convolve_sequence_gradient, max_pool, max_pool_gradient
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from sequence_convolution.models import SequenceConvolutionNet, \
     SequenceConvolutionLayer, MultiSequenceConvolutionLayer, MaxPoolingLayer
from sequence_convolution.seq_array import SeqArray, sample_sequence
from neural_nets.data_providers import MiniBatchDataProvider
from neural_nets.optimizers import SGD
from neural_nets.schedulers import constant_scheduler
from neural_nets.parameter_updaters import SimpleSGDUpdate
from neural_nets.monitors import SimpleProgressMonitor
from copy import copy, deepcopy
from itertools import izip

STRIDE = 4

def checkgrad_model(layer, input, epsilon=1e-4, **kwargs):
    cache = layer.feed_forward(input)
    f0 = np.sum(cache[0].get())
    df_output = gpuarray.empty_like(cache[0]).fill(1.)
    grads = layer.backprop(input, df_output, cache)[0]
    dtype = grads[0].dtype
    eps = np.finfo(dtype).eps

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

        loss += np.sum(((grads[i].get() - grad_approx_i) / (grads[i].get() + eps)) ** 2.)
    loss = np.sqrt(loss)

    return loss

class TestConvolution(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-4
    DOUBLE_ERR_TOL = 1e-13

    @staticmethod
    def cpu_conv1d(x, w, b):
        height, width = x.shape        
        n_filters= w.shape[0]
        filter_width = w.shape[1] // 4

        pad_width = filter_width - 1
        x_padded = np.concatenate((x,
                                   np.zeros((height, pad_width), dtype=np.int8)), 1)

        y = np.empty((height, n_filters, width), dtype=w.dtype)

        for f in range(n_filters):
            for j in range(width):
                y[:, f, j] = b[f]
                for k in range(filter_width):
                    nt = x_padded[:, j + k]
                    y[np.bool_(nt & DNA_A),f,j] += w[f,4*k]
                    y[np.bool_(nt & DNA_C),f,j] += w[f,4*k+1]
                    y[np.bool_(nt & DNA_G),f,j] += w[f,4*k+2]
                    y[np.bool_(nt & DNA_T),f,j] += w[f,4*k+3]
        return y.reshape(height, n_filters*width)

    @staticmethod    
    def gpu_conv1d(x, w, b, dtype):
        y = convolve_sequence(x, w, b)
        return y
    
    def conv1d_test_setup(self, height, width, filter_width, n_filters):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL), 
                               (np.float64, self.DOUBLE_ERR_TOL)):
            seq = [''.join((random.choice('ACGT') for i in range(width)))
                   for j in range(height)]
            sa = SeqArray(seq)
            x = sa.enc_seq
            w = curand((n_filters, 4*filter_width), dtype=dtype)
            b = curand((n_filters,), dtype=dtype)
            y = self.gpu_conv1d(x, w, b, dtype)
            y_np = self.cpu_conv1d(x.get(), w.get(), b.get())
            y_cpu = y.get()
            
            self.assertLess(np.max((y_cpu - y_np) / y_cpu), err_tol)
    
    def test_conv1d_matrix_small(self):
        for i in range(20):
            n = np.random.randint(4, 9)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)

            self.conv1d_test_setup(n, m, 1, n_filters)

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

    def test_input_offset(self):
        for i in range(20):
            height = np.random.randint(10, 100)
            width = np.random.randint(10, 300) 
            n_filters = np.random.randint(2, 10)
            total_width = np.random.randint(width + 1, width+100)
            input_offset = np.random.randint(0, total_width-width)
            total_target_width = np.random.randint(n_filters*width+1,
                                                   n_filters*width+200)
            target_offset = np.random.randint(0, 
                                              total_target_width-n_filters*width)
            filter_width = np.random.randint(2, 24)

            for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL), 
                                   (np.float64, self.DOUBLE_ERR_TOL)):
                seq = [''.join((random.choice('ACGT') for i in range(width)))
                       for j in range(height)]
                sa = SeqArray(seq)
                x = sa.enc_seq.get()
                X = np.empty((height, total_width), dtype=np.int8)
                X[:,input_offset:input_offset+width] = x
                X = gpuarray.to_gpu(X)
                
                w = curand((n_filters, 4*filter_width), dtype=dtype)
                b = curand((n_filters,), dtype=dtype)
                y = gpuarray.empty((X.shape[0], total_target_width), dtype)
                
                y = convolve_sequence(X, w, b, input_offset, target_offset,
                                      width, y)

                y_np = self.cpu_conv1d(x, w.get(), b.get())
                y_cpu = y.get()[:,target_offset:target_offset+n_filters*width]

                self.assertLess(np.max((y_cpu - y_np) / y_cpu), err_tol)

class TestConvolutionGradWeights(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-3
    DOUBLE_ERR_TOL = 1e-12
    
    @staticmethod
    def grad_weights_cpu(input, df_output, n_filters, filter_width):
        df_output = df_output.reshape((input.shape[0], 
                                       n_filters, input.shape[1]))
        height, width = input.shape
        df_w = np.zeros((n_filters, 4*filter_width))
        input_padded = np.concatenate((input, 
                                       np.zeros((input.shape[0], 
                                                 filter_width-1),
                                                 dtype=np.int8)), 1)
        for n in range(n_filters):
            for i in range(filter_width):
                df_w[n, 4*i] += df_output[:,n,:][
                    np.bool_(input_padded[:,i:i+width] & DNA_A)].sum()
                  
                df_w[n, 4*i+1] += df_output[:,n,:][
                    np.bool_(input_padded[:,i:i+width] & DNA_C)].sum()
                  
                df_w[n, 4*i+2] += df_output[:,n,:][
                    np.bool_(input_padded[:,i:i+width] & DNA_G)].sum()
                  
                df_w[n, 4*i+3] += df_output[:,n,:][
                    np.bool_(input_padded[:,i:i+width] & DNA_T)].sum()
                  
        return df_w

    def grad_weights_test(self, height, width, n_filters, filter_width):
        for dtype, err_tol in ((np.float64, self.DOUBLE_ERR_TOL),
                               (np.float32, self.FLOAT_ERR_TOL)):

            seq = [''.join((random.choice('ACGT') for i in range(width)))
                   for j in range(height)]
            sa = SeqArray(seq)
            eps = np.finfo(dtype).eps
            x = sa.enc_seq
            df_output = curand((height, n_filters*width), dtype)

            df_w = convolve_sequence_gradient(x, df_output, filter_width, 
                                              n_filters, block_size=1024)
            df_w_cpu = df_w.get()
            df_w_np = self.grad_weights_cpu(x.get(), 
                                            df_output.get(), 
                                            n_filters, 
                                            filter_width)

            self.assertLess(np.abs((df_w_cpu-df_w_np)/(df_w_cpu+eps)).max(), err_tol)

    def test_grad_weights_filter_12(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = 12
            m = np.random.randint(filter_width, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_16(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = 16
            m = np.random.randint(filter_width, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_24(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = 24
            m = np.random.randint(filter_width, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_8(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = 8
            m = np.random.randint(filter_width, 100)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_filter_4(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = 4
            m = np.random.randint(filter_width, 200)
            n_filters = np.random.randint(2, 100)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights(self):
        for i in range(20):
            n = np.random.randint(5, 300)
            filter_width = np.random.randint(2, 36)
            m = np.random.randint(filter_width, 500)
            n_filters = np.random.randint(2, 50)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_1_block(self):
        for i in range(20):
            n = 10
            m = 16
            filter_width = 1
            n_filters = 5
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_input_offset(self):
        for i in range(20):
            height = np.random.randint(10, 100)
            width = np.random.randint(10, 300)
            total_width = np.random.randint(width + 1, width + 100)
            input_offset = np.random.randint(0, total_width - width)            
            filter_width = np.random.randint(2, 24)
            n_filters = np.random.randint(2, 10)
            df_output_width = n_filters * width
            total_df_output_width = np.random.randint(df_output_width+1, 
                                                      df_output_width+1000)
            df_output_offset = np.random.randint(0,
                                                 total_df_output_width-df_output_width)
        
            for dtype, err_tol in ((np.float64, self.DOUBLE_ERR_TOL),
                                   (np.float32, self.FLOAT_ERR_TOL)):

                seq = [''.join((random.choice('ACGT') for i in range(width)))
                       for j in range(height)]
                sa = SeqArray(seq)
                eps = np.finfo(dtype).eps
                x = sa.enc_seq.get()
                X = np.empty((height, total_width), dtype=np.int8)
                X[:,input_offset:input_offset+width] = x
                X = gpuarray.to_gpu(X)
                
                df_output = curand((height, total_df_output_width), dtype)
                

                df_w = convolve_sequence_gradient(X, df_output, filter_width, n_filters, 
                                                  input_offset, df_output_offset, width,
                                                  block_size=1024)
                df_w_cpu = df_w.get()
                df_output_cpu = df_output.get()[:,df_output_offset:df_output_offset+df_output_width]
                df_w_np = self.grad_weights_cpu(x, 
                                                df_output_cpu,
                                                n_filters, filter_width)

                self.assertLess(np.abs((df_w_cpu-df_w_np)/(df_w_cpu+eps)).max(), err_tol)

class TestMaxPool(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_cpu(mat, pool_size, n_filters):
        mat = mat.reshape((mat.shape[0], n_filters, mat.shape[1] / n_filters))
        width_pooled = int(np.ceil(mat.shape[2] / float(pool_size)))
        width_pad = (width_pooled * pool_size) % mat.shape[2]
        mat_padded = np.concatenate((mat, 
                                     -np.inf * np.ones((mat.shape[0], mat.shape[1],
                                                        width_pad),
                                                        mat.dtype)), 2)
                                    
        output = np.empty((mat.shape[0], mat.shape[1], width_pooled))
        for n in range(output.shape[0]):
            for i in range(output.shape[2]):
                output[n,:,i] = np.max(mat_padded[n,:, pool_size*i:pool_size*(i+1)], 1)

        return output.reshape((mat.shape[0], mat.shape[1]*width_pooled))

    def max_pool_test(self, height, width, n_filters, pool_size):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                               (np.float64, self.DOUBLE_ERR_TOL)):

            mat = curand((height, n_filters*width), dtype)
            target, argmax = max_pool(mat, pool_size, n_filters)
            target_cpu = target.get()
            target_np = self.max_pool_cpu(mat.get(), pool_size, n_filters)
            self.assertLess(np.linalg.norm(
                (target_cpu - target_np) / target_cpu, np.inf), 
                err_tol)

    def test_max_pool(self):
        for i in range(20):
            height = np.random.randint(100, 1000)
            width = np.random.randint(20, 500)
            n_filters = np.random.randint(2, 10)
            pool_size = np.random.randint(2, width)
            self.max_pool_test(height, width, n_filters, pool_size)

    def test_max_pool_offset(self):
        for i in range(20):
            height = np.random.randint(100, 1000)
            width = np.random.randint(20, 500)
            n_filters = np.random.randint(2, 10)
            total_width = np.random.randint(n_filters*width+1, 
                                            n_filters*width+300)
            pool_size = np.random.randint(2, width)
            pooled_width = int(np.ceil(width / float(pool_size)))
            input_offset = np.random.randint(0, total_width-n_filters*width)
            total_width_pooled = np.random.randint(n_filters*pooled_width+1, 
                                                   n_filters*pooled_width+100)            
            pooled_offset = np.random.randint(0, total_width_pooled-n_filters*pooled_width)

            for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                                   (np.float64, self.DOUBLE_ERR_TOL)):

                mat = curand((height, total_width), dtype)
                target = gpuarray.empty((height, total_width_pooled), dtype)
                argmax = gpuarray.empty(target.shape, np.uint32)
                target, argmax = max_pool(mat, pool_size, n_filters, width,
                                          input_offset, pooled_offset,
                                          target, argmax)
                target_cpu = target.get()[:,pooled_offset:pooled_offset+n_filters*pooled_width]
                target_np = self.max_pool_cpu(mat.get()[:,input_offset:input_offset+n_filters*width], 
                                              pool_size, n_filters)
                self.assertLess(np.linalg.norm(
                    (target_cpu - target_np) / target_cpu, np.inf), 
                    err_tol)

class TestMaxPoolGradient(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_grad_cpu(mat, mat_pooled, argmax, df_output, pool_size, n_filters):
        height = mat.shape[0]
        width = mat.shape[1] / n_filters
        width_pooled = mat_pooled.shape[1] / n_filters
        mat = mat.reshape((height, n_filters, width))
        mat_pooled = mat_pooled.reshape((height, n_filters, width_pooled))
        argmax = argmax.reshape(mat_pooled.shape)
        df_output = df_output.reshape(mat_pooled.shape)
        
        df_input = np.zeros_like(mat)
        for n in range(n_filters):
            for i in range(width_pooled):
                df_input[range(height), height*[n], argmax[:, n, i]] = df_output[:, n, i]
        
        return df_input.reshape((height, n_filters*width))

    def max_pool_grad_test(self, height, width, n_filters, pool_size):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                               (np.float64, self.DOUBLE_ERR_TOL)):
            mat = curand((height, n_filters*width), dtype)
            mat_pooled, argmax = max_pool(mat, pool_size, n_filters)
            df_output = curand(mat_pooled.shape, dtype)
            df_input = max_pool_gradient(mat, argmax, df_output, 
                                         pool_size, n_filters)
            df_input_cpu = df_input.get()
            df_input_np = self.max_pool_grad_cpu(mat.get(), mat_pooled.get(), 
                                                 argmax.get(),
                                                 df_output.get(), pool_size,
                                                 n_filters)

            self.assertTrue(np.all(df_input_cpu == df_input_np))

    def test_max_pool_grad(self):
        for i in range(20):
            n = np.random.randint(100, 1000)
            m = np.random.randint(15, 500)
            n_filters = np.random.randint(2, 10)
            pool_size = np.random.randint(2, 15)
            self.max_pool_grad_test(n, m, n_filters, pool_size)

    def test_max_pool_grad_offset(self):
        for i in range(20):
            height = np.random.randint(100, 1000)
            width = np.random.randint(15, 500)
            n_filters = np.random.randint(2, 10)
            total_width = np.random.randint(n_filters*width+1, n_filters*width+300)
            input_offset = np.random.randint(0, total_width-n_filters*width)
            pool_size = np.random.randint(2, width)
            pooled_width = int(np.ceil(width / float(pool_size)))
            total_width_pooled = np.random.randint(n_filters*pooled_width+1, 
                                                   n_filters*pooled_width+100)
            pooled_offset = np.random.randint(0, total_width_pooled-n_filters*pooled_width)

            for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),
                                   (np.float64, self.DOUBLE_ERR_TOL)):
                mat = curand((height, n_filters*total_width), dtype)
                mat_pooled = gpuarray.empty((height, n_filters*total_width_pooled), dtype)
                argmax = gpuarray.empty(mat_pooled.shape, np.uint32)
                mat_pooled, argmax = max_pool(mat, pool_size, n_filters,  width,
                                              input_offset, pooled_offset,
                                              mat_pooled, argmax)
                df_output = curand(mat_pooled.shape, dtype)
                df_input = max_pool_gradient(mat, argmax, df_output, pool_size,
                                             n_filters,
                                             width, pooled_width,
                                             input_offset, pooled_offset)
                df_input_cpu = df_input.get()[:,input_offset:input_offset+n_filters*width]
                mat_cpu = mat.get()[:,input_offset:input_offset+n_filters*width]
                mat_pooled_cpu = mat_pooled.get()[:,pooled_offset:pooled_offset+n_filters*pooled_width]
                argmax_cpu = argmax.get()[:,pooled_offset:pooled_offset+n_filters*pooled_width]
                df_output_cpu = df_output.get()[:,pooled_offset:pooled_offset+n_filters*pooled_width]
                df_input_np = self.max_pool_grad_cpu(mat_cpu, mat_pooled_cpu, 
                                                     argmax_cpu,
                                                     df_output_cpu, pool_size, n_filters)

                self.assertTrue(np.all(df_input_cpu == df_input_np))
            


class TestConvNet(unittest.TestCase):
    def test_conv_net(self):
        seq = ['A' + ''.join([random.choice('ACGT') for x in range(7)]) for x in range(100)] + \
          ['T' + ''.join([random.choice('ACGT') for x in range(7)]) for x in range(100)]
        targets = np.array(100*[[1., 0.]] + 100*[[0., 1.]], dtype=np.float32)

        shuffle_idx = np.random.permutation(len(seq))
        seq = [seq[i] for i in shuffle_idx]
        targets = gpuarray.to_gpu(targets[shuffle_idx])

        sa = SeqArray(seq)
        test_error = 1
        for i in range(10):
            model = SequenceConvolutionNet(sa.enc_seq.shape[1], 2, 32, 5, 8, [], 
                                           activation_function='tanh')
            
            train_data = MiniBatchDataProvider(sa.enc_seq, targets, 10)

            optimizer = SGD(model, SimpleSGDUpdate, train_data, train_data,
                            learning_rate_schedule=constant_scheduler(1.),
                            progress_monitor=SimpleProgressMonitor())

            optimizer.run(20)
            test_error = np.min([optimizer.best_validation_loss, test_error])

        self.assertEqual(test_error, 0.)

class TestConvolutionGradient(unittest.TestCase):
    EPSILON = 1e-2
    TOL = 1e-4
    
    def test_convolution_gradient(self):
        for i in range(20):
            n_in = 36
            filter_width = 12
            n_filters = 4
            conv_layer = SequenceConvolutionLayer(n_in, filter_width, n_filters, 
                                                  dtype=np.float64)

            seq = [''.join((random.choice('ACGT'))) for i in range(n_in)
                   for j in range(100)]
            sa = SeqArray(seq)
            loss = checkgrad_model(conv_layer, sa.enc_seq, epsilon=self.EPSILON)
            self.assertLess(loss, self.TOL)

class TestMultiSequenceConvolutionLayer(unittest.TestCase):
    """ Test whether what MultiSequenceConvolutionLayer is doing is identical
    to SequenceConvolutionLayer 
    """
    
    N = 100
    multi_conv_config = [{'n_in': 50, 'n_filters': 10, 'filter_width': 5, 
              'activation_function': 'tanh', 'pool_size': 5},
             {'n_in': 50, 'weight_share': 0, 'pool_size': 2},
             {'n_in': 100, 'n_filters': 12, 'filter_width': 10, 
              'activation_function': 'tanh', 'pool_size': 8}]
    

    def setUp(self):
        sa = [sample_sequence(conf['n_in'], self.N) for conf in
              self.multi_conv_config]
        self.input = [s.enc_seq for s in sa]

        # Create multi-convolution layer
        self.conv_layer_multi = MultiSequenceConvolutionLayer(self.multi_conv_config)


        # Convert configuration to single convolution
        single_conv_config = deepcopy(self.multi_conv_config)
        single_conv_config[1]['n_filters'] = single_conv_config[0]['n_filters']
        single_conv_config[1]['filter_width'] = single_conv_config[0]['filter_width']
        single_conv_config[1]['activation_function'] = \
          single_conv_config[0]['activation_function']
        self.single_conv_config = single_conv_config

        # Create single convolution layers
        self.conv_layers_single = [SequenceConvolutionLayer(
            conf['n_in'], conf['filter_width'],
            conf['n_filters'], 
            conf['activation_function']) 
            for conf in single_conv_config]

        # Weight-sharing
        self.conv_layers_single[0].parameters = \
          (self.conv_layer_multi.W[0], self.conv_layer_multi.b[0])
        self.conv_layers_single[1].parameters = \
          (self.conv_layer_multi.W[0], self.conv_layer_multi.b[0])        
        self.conv_layers_single[2].parameters = \
          (self.conv_layer_multi.W[1], self.conv_layer_multi.b[1])        
        
        self.maxpool_layers_single = [MaxPoolingLayer(conf['n_in'], conf['pool_size'],
                                                      conf['n_filters'])
                                                      for conf in self.single_conv_config]


    def test_feed_forward(self):
        activations_multi, argmax, filtermaps = \
          self.conv_layer_multi.feed_forward(self.input)

        filtermaps_single = []
        argmax_single = []
        activations_single = []
        for layer_conv, layer_pool, input_single \
          in izip(self.conv_layers_single, self.maxpool_layers_single, self.input):
            filtermap, = layer_conv.feed_forward(input_single)
            filtermaps_single.append(filtermap)
            activations, argmax = layer_pool.feed_forward(filtermap)
            argmax_single.append(argmax)
            activations_single.append(activations)

        activations_joined = np.concatenate([a.get() for a in activations_single], 1)

        self.assertEqual(np.abs(activations_multi.get() - activations_joined).max(), 0.)

    def test_backprop(self):
        activations_multi, argmax_multi, filtermaps_multi = \
          self.conv_layer_multi.feed_forward(self.input)

        df_output_cpu = [np.asarray(np.random.rand(self.N, l.n_units), 
                                    dtype=activations_multi.dtype)
                         for l in self.maxpool_layers_single]
        df_output_single = map(gpuarray.to_gpu, df_output_cpu)
        df_output_multi = gpuarray.to_gpu(
            np.ascontiguousarray(np.concatenate(df_output_cpu, 1)))

        grads_multi_conv, df_filtermaps_multi = \
          self.conv_layer_multi.backprop(
              self.input, df_output_multi,
              cache=(activations_multi, argmax_multi, filtermaps_multi))

        filtermaps_single = []
        argmax_single = []
        activations_single = []
        df_W_single = []
        df_b_single = []
        for i, (layer_conv, layer_pool, input_single, df_o) \
          in enumerate(izip(self.conv_layers_single, 
                            self.maxpool_layers_single, 
                            self.input, df_output_single)):
            filtermap, = layer_conv.feed_forward(input_single)
            filtermaps_single.append(filtermap)
            
            activations, argmax = layer_pool.feed_forward(filtermap)
            argmax_single.append(argmax)
            activations_single.append(activations)

            _, df_filtermap = layer_pool.backprop(filtermap, df_o,
                                                  cache=(activations, argmax))
            (df_W_layer, df_b_layer), _ = layer_conv.backprop(
                input_single, df_filtermap, (filtermap,))

            if i in (0, 2):
                df_W_single.append(df_W_layer)
                df_b_single.append(df_b_layer)
            elif i == 1:
                df_W_single[0] += df_W_layer
                df_b_single[0] += df_b_layer

        grads_single = df_W_single + df_b_single

        for g_multi, g_single in izip(grads_multi_conv, grads_single):
            if g_multi is None and g_single is None: continue
            self.assertEqual(np.abs(g_multi.get() - g_single.get()).max(), 0.)

if __name__ == '__main__':
    unittest.main()
