# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import unittest
import random
import numpy as np
import hebel
if not hebel.is_initialized:
    hebel.init()

from pycuda import driver
from sequence_convolution.pycuda_ops import convolve_sequence, \
     convolve_sequence_gradient, max_pool, max_pool_gradient
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from sequence_convolution.seq_array import encode_sequence, sample_sequence
from sequence_convolution.models import SequenceConvolutionNet, \
     SequenceConvolutionLayer, MultiSequenceConvolutionLayer, MaxPoolingLayer
from sequence_convolution.seq_array import SeqArrayDataProvider, sample_sequence, \
    encode_sequence
from hebel.data_providers import MiniBatchDataProvider
from hebel.optimizers import SGD
from hebel.schedulers import constant_scheduler
from hebel.parameter_updaters import SimpleSGDUpdate
from hebel.monitors import SimpleProgressMonitor
from copy import copy, deepcopy
from itertools import izip

def checkgrad_model(layer, input_data, epsilon=1e-4, **kwargs):
    cache = layer.feed_forward(input_data)
    f0 = np.sum(cache[0].get())
    df_output = gpuarray.empty_like(cache[0]).fill(1.)
    grads = layer.backprop(input_data, df_output, cache)[0]
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
            f1 = np.sum(layer.feed_forward(input_data)[0].get())

            # Get f(x + epsilon)
            param_i[idx] -= 2 * epsilon
            p[i] = gpuarray.to_gpu(param_i)
            layer.parameters = p
            f2 = np.sum(layer.feed_forward(input_data)[0].get())

            # Reset weight
            param_i[idx] = w0
            p[i] = gpuarray.to_gpu(param_i)
            layer.parameters = p

            # Compute gradient approximation
            grad_approx_i[idx] = (f1 - f2) / (2 * epsilon)

        loss += np.sum(((grads[i].get() - grad_approx_i) /
                        (grads[i].get() + eps)) ** 2.)
    loss = np.sqrt(loss) / np.sum([g.size for g in grads])

    return loss


class TestConvolution(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-4
    DOUBLE_ERR_TOL = 1e-13

    @staticmethod
    def cpu_conv1d(x, w, b):
        height, width = x.shape
        n_filters = w.shape[0]
        filter_width = w.shape[1] // 4
        output_width = width - filter_width + 1

        y = np.empty((height, n_filters, output_width), dtype=w.dtype)

        for f in range(n_filters):
            for j in range(output_width):
                y[:, f, j] = b[f]
                for k in range(filter_width):
                    nt = x[:, j + k]
                    y[np.bool_(nt == 'A'), f, j] += w[f, 4 * k]
                    y[np.bool_(nt == 'C'), f, j] += w[f, 4 * k + 1]
                    y[np.bool_(nt == 'G'), f, j] += w[f, 4 * k + 2]
                    y[np.bool_(nt == 'T'), f, j] += w[f, 4 * k + 3]
                    y[np.bool_(nt == 'R'), f, j] += \
                        .5 * w[f, 4 * k] + .5 * w[f, 4 * k + 2]
                    y[np.bool_(nt == 'Y'), f, j] += \
                        .5 * w[f, 4 * k + 1] + .5 * w[f, 4 * k + 3]
        y = np.rollaxis(y, 1, 3)
        return y.reshape(height, n_filters * output_width)

    @staticmethod
    def gpu_conv1d(x, w, b):
        y = convolve_sequence(x, w, b)
        return y

    def conv1d_test_setup(self, height, width, filter_width, n_filters):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),):
                               # (np.float64, self.DOUBLE_ERR_TOL)):
            seq = sample_sequence(width, height)
            x = gpuarray.to_gpu(encode_sequence(seq))
            w = gpuarray.to_gpu(np.random.rand(n_filters, 4 * filter_width).astype(dtype))
            b = gpuarray.to_gpu(np.random.rand(n_filters).astype(dtype))
            y_np = self.cpu_conv1d(x.get(), w.get(), b.get())
            y = self.gpu_conv1d(x, w, b)
            y_cpu = y.get()

            if not y_cpu.shape == y_np.shape:
                import pudb; pudb.set_trace()
            self.assertLess(np.max((y_cpu - y_np) / y_cpu), err_tol)
            del x, w, b, y

    def test_conv1d_matrix_small(self):
        for _ in range(20):
            n = np.random.randint(10, 100)
            m = np.random.randint(100, 200)
            n_filters = np.random.randint(2, 50)
            filter_width = np.random.choice([4, 8, 16, 32])

            self.conv1d_test_setup(n, m, filter_width, n_filters)

    def test_conv1d_matrix_big_height(self):
        for _ in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(4, 9)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big_width(self):
        for _ in range(20):
            n = np.random.randint(4, 9)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big(self):
        for _ in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, 4, n_filters)

    def test_conv1d_matrix_big_filter(self):
        for _ in range(20):
            n = np.random.randint(200, 1000)
            m = np.random.randint(200, 1000)
            w = 2 * np.random.randint(2, 5)
            n_filters = np.random.randint(2, 5)
            self.conv1d_test_setup(n, m, w, n_filters)


class TestConvolutionGradWeights(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-3
    DOUBLE_ERR_TOL = 1e-12

    @staticmethod
    def grad_weights_cpu(input_data, df_output, n_filters, filter_width):
        height, width = input_data.shape
        output_width = width - filter_width + 1
        df_output = df_output.reshape((input_data.shape[0],
                                       output_width,
                                       n_filters))
        df_w = np.zeros((n_filters, filter_width, 4))

        for n in range(n_filters):
            for i in range(filter_width):
                df_w[n, i, 0] += df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'A')].sum() + \
                    .5 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'R')].sum() + \
                    .25 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'N')].sum()

                df_w[n, i, 1] += df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'C')].sum() + \
                    .5 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'Y')].sum() + \
                    .25 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'N')].sum()

                df_w[n, i, 2] += df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'G')].sum() + \
                    .5 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'R')].sum() + \
                    .25 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'N')].sum()

                df_w[n, i, 3] += df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'T')].sum() + \
                    .5 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'Y')].sum() + \
                    .25 * df_output[:, :, n][
                    np.bool_(input_data[:, i:i+output_width] == 'N')].sum()

        df_w = df_w.reshape((df_w.shape[0], df_w.shape[1] * df_w.shape[2]))
        return df_w

    def grad_weights_test(self, height, width, n_filters, filter_width):
        for dtype, err_tol in (# (np.float64, self.DOUBLE_ERR_TOL),
                               (np.float32, self.FLOAT_ERR_TOL),):

            output_width = width - filter_width + 1
            eps = np.finfo(dtype).eps
            x = gpuarray.to_gpu(encode_sequence(sample_sequence(width, height)))
            df_output = gpuarray.to_gpu(
                np.random.rand(height, output_width, n_filters).astype(dtype))

            df_w = convolve_sequence_gradient(x, df_output, filter_width,
                                              n_filters)
            df_w_cpu = df_w.get()
            df_w_np = self.grad_weights_cpu(x.get(),
                                            df_output.get(),
                                            n_filters,
                                            filter_width)

            self.assertLess(np.abs((df_w_cpu - df_w_np) /
                                   (df_w_cpu + eps)).max(), err_tol)

    def test_grad_weights(self):
        for _ in range(20):
            n = np.random.randint(20, 200)
            filter_width = np.random.randint(8, 32)
            m = np.random.randint(filter_width, 200)
            n_filters = np.random.randint(2, 12)
            self.grad_weights_test(n, m, n_filters, filter_width)

    def test_grad_weights_small(self):
        for _ in range(20):
            n = np.random.randint(20, 100)
            filter_width = np.random.randint(4, 16)
            m = np.random.randint(filter_width, 32)
            n_filters = np.random.randint(2, 12)
            self.grad_weights_test(n, m, n_filters, filter_width)

class TestMaxPool(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_cpu(x, pooling_size, n_filters):
        height = x.shape[0]
        input_width = x.shape[1] / n_filters
        output_width = input_width // pooling_size
        y = x.reshape((height, output_width, pooling_size, n_filters))\
             .max(2)\
             .reshape((height, n_filters * output_width))
        return y
    # def max_pool_cpu(mat, pool_size):
    #     assert not mat.shape[1] % pool_size
    #     width_pooled = mat.shape[1] // pool_size

    #     output = mat.reshape((mat.shape[0]*width_pooled, pool_size))\
    #                 .max(1).reshape((mat.shape[0], width_pooled))

    #     return output

    def max_pool_test(self, height, width, pool_size, n_filters):
        for dtype, err_tol in ((np.float32, self.FLOAT_ERR_TOL),):
                               # (np.float64, self.DOUBLE_ERR_TOL)):

            mat = gpuarray.to_gpu(np.random.rand(height, width * n_filters)
                                  .astype(dtype))
            target, argmax = max_pool(mat, pool_size, n_filters)
            target_cpu = target.get()
            target_np = self.max_pool_cpu(mat.get(), pool_size, n_filters)
            self.assertLess(np.linalg.norm(
                (target_cpu - target_np) / target_cpu, np.inf),
                err_tol)
            del mat, target, argmax

    def test_max_pool(self):
        for _ in range(20):
            height = np.random.randint(100, 1000)
            pool_size = np.random.randint(2, 64)
            width = pool_size * np.random.randint(20, 300)
            n_filters = np.random.randint(2, 64)
            self.max_pool_test(height, width, pool_size, n_filters)


class TestMaxPoolGradient(unittest.TestCase):
    FLOAT_ERR_TOL = 1e-20
    DOUBLE_ERR_TOL = 1e-20

    @staticmethod
    def max_pool_grad_cpu(mat, mat_pooled, argmax,
                          df_output, pool_size):
        height = mat.shape[0]
        width = mat.shape[1]
        width_pooled = mat_pooled.shape[1]

        df_input = np.zeros_like(mat).reshape((height*width_pooled, pool_size))
        df_input[np.arange(argmax.size), argmax.ravel()] = df_output.ravel()
        
        return df_input.reshape(mat.shape)

    @unittest.skip("Not implemented")
    def max_pool_grad_test(self, height, width, pool_size):
        for dtype in (np.float32, np.float64):
            mat = gpuarray.to_gpu(np.random.rand(height, width).astype(dtype))
            mat_pooled, argmax = max_pool(mat, pool_size)
            df_output = gpuarray.to_gpu(np.random.rand(*mat_pooled.shape).astype(dtype))
            df_input = max_pool_gradient(mat, argmax, df_output, pool_size)
            df_input_cpu = df_input.get()
            df_input_np = self.max_pool_grad_cpu(mat.get(), mat_pooled.get(),
                                                 argmax.get(),
                                                 df_output.get(), pool_size)
            self.assertTrue(np.all(df_input_cpu == df_input_np))
            del mat, mat_pooled, df_output, df_input, argmax

    @unittest.skip("Not implemented")
    def test_max_pool_grad(self):
        for _ in range(20):
            n = np.random.randint(10, 1000)
            pool_size = np.random.randint(2, 64)
            m = np.random.randint(10, 500) * pool_size
            self.max_pool_grad_test(n, m, pool_size)

    @unittest.skip("Not implemented")
    def test_max_pool_grad_offset(self):
        for _ in range(20):
            height = np.random.randint(100, 1000)
            pool_size = np.random.randint(2, 64)
            width = np.random.randint(10, 500) * pool_size

            total_width = np.random.randint(
                width + 1, width + 300)
            input_offset = np.random.randint(
                0, total_width - width)

            pooled_width = width // pool_size
            total_width_pooled = np.random.randint(
                pooled_width + 1,
                pooled_width + 100)
            pooled_offset = np.random.randint(
                0, total_width_pooled - pooled_width)

            for dtype in (np.float32, np.float64):
                mat = gpuarray.to_gpu(np.random.rand(height, total_width)
                                      .astype(dtype))
                mat_pooled = gpuarray.empty(
                    (height, total_width_pooled), dtype)
                argmax = gpuarray.empty(mat_pooled.shape, np.uint32)
                mat_pooled, argmax = max_pool(mat, pool_size,
                                              width,
                                              input_offset, pooled_offset,
                                              mat_pooled, argmax)
                df_output = gpuarray.to_gpu(np.random.rand(*mat_pooled.shape).astype(dtype))
                df_input = gpuarray.empty_like(mat).fill(-99)
                df_input = max_pool_gradient(mat, argmax, df_output, pool_size,
                                             width, pooled_width,
                                             input_offset, pooled_offset, target=df_input)
                df_input_cpu = \
                    df_input.get()[:,
                        input_offset:input_offset + width]
                mat_cpu = \
                    mat.get()[:, input_offset:input_offset + width]
                mat_pooled_cpu = \
                    mat_pooled.get()[:,
                        pooled_offset:pooled_offset + pooled_width]
                argmax_cpu = argmax.get()[:,
                    pooled_offset:pooled_offset + pooled_width]
                df_output_cpu = df_output.get()[:,
                    pooled_offset:pooled_offset + pooled_width]
                df_input_np = self.max_pool_grad_cpu(mat_cpu, mat_pooled_cpu,
                                                     argmax_cpu,
                                                     df_output_cpu,
                                                     pool_size)

                self.assertTrue(np.all(df_input_cpu == df_input_np))
                del mat, mat_pooled, df_output, df_input, argmax


class TestConvNet(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_conv_net(self):
        seq = ['A' + ''.join([random.choice('ACGT') for _ in range(7)])
               for _ in range(100)] + \
              ['T' + ''.join([random.choice('ACGT') for _ in range(7)])
               for _ in range(100)]
        targets = np.array(100 * [[1., 0.]] +
                           100 * [[0., 1.]], dtype=np.float32)

        shuffle_idx = np.random.permutation(len(seq))
        seq = [seq[i] for i in shuffle_idx]
        targets = gpuarray.to_gpu(targets[shuffle_idx])

        test_error = 1
        train_data = SeqArrayDataProvider(seq, targets, 10)

        for _ in range(10):
            model = SequenceConvolutionNet(
                train_data.enc_seq.shape[1], 2, 32, 5, 8, [],
                activation_function='tanh')

            optimizer = SGD(model, SimpleSGDUpdate, train_data, train_data,
                            learning_rate_schedule=constant_scheduler(1.),
                            progress_monitor=SimpleProgressMonitor())

            optimizer.run(20)
            test_error = np.min([optimizer.best_validation_loss, test_error])

        self.assertEqual(test_error, 0.)


class TestConvolutionGradient(unittest.TestCase):
    EPSILON = 1e-2
    TOL = 1e-3

    @unittest.skip("Not implemented")
    def test_convolution_gradient(self):
        for _ in range(20):
            n_in = 36
            filter_width = 12
            n_filters = 4
            conv_layer = SequenceConvolutionLayer(
                n_in, filter_width, n_filters,
                dtype=np.float64)

            seq = [''.join((random.choice('ACGT') for i in range(n_in)))
                   for _ in range(100)]
            x = gpuarray.to_gpu(encode_sequence(seq))
            loss = checkgrad_model(conv_layer,
                                   x, epsilon=self.EPSILON)
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
        seq = [sample_sequence(conf['n_in'], self.N) for conf in
              self.multi_conv_config]
        self.input = [gpuarray.to_gpu(encode_sequence(s)) for s in seq]

        # Create multi-convolution layer
        self.conv_layer_multi = MultiSequenceConvolutionLayer(
            self.multi_conv_config)

        # Convert configuration to single convolution
        single_conv_config = deepcopy(self.multi_conv_config)
        single_conv_config[1]['n_filters'] = single_conv_config[0]['n_filters']
        single_conv_config[1]['filter_width'] = \
            single_conv_config[0]['filter_width']
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

        self.maxpool_layers_single = \
            [MaxPoolingLayer(conf['n_in'], conf['pool_size'],
                             conf['n_filters'])
             for conf in self.single_conv_config]

    @unittest.skip("Not implemented")
    def test_feed_forward(self):
        activations_multi, argmax, filtermaps, dropout_mask, activations_fc = \
          self.conv_layer_multi.feed_forward(self.input)

        filtermaps_single = []
        argmax_single = []
        activations_single = []
        for layer_conv, layer_pool, input_single \
          in izip(self.conv_layers_single,
                  self.maxpool_layers_single, self.input):

            filtermap, = layer_conv.feed_forward(input_single)
            filtermaps_single.append(filtermap)
            activations, argmax = layer_pool.feed_forward(filtermap)
            argmax_single.append(argmax)
            activations_single.append(activations)

        activations_joined = np.concatenate(
            [a.get() for a in activations_single], 1)

        self.assertEqual(
            np.abs(activations_multi.get() - activations_joined).max(), 0.)

    @unittest.skip("Not implemented")
    def test_backprop(self):
        activations_multi, argmax_multi, filtermaps_multi, \
        dropout_mask, activations_fc = \
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
              cache=(activations_multi, argmax_multi,
                     filtermaps_multi, dropout_mask, activations_fc))

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
