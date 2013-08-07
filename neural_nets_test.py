import unittest
import random
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.curandom import rand as curand
import pycuda.driver as drv
from neural_nets.models import NeuralNet
from neural_nets.optimizers import SGD
from neural_nets.parameter_updaters import SimpleSGDUpdate, MomentumUpdate, NesterovMomentumUpdate
from neural_nets.data_providers import MiniBatchDataProvider, BatchDataProvider, MNISTDataProvider
from neural_nets.schedulers import constant_scheduler, exponential_scheduler, linear_scheduler_up
from neural_nets.pycuda_ops.matrix import extract_columns

class TestNeuralNetMNIST(unittest.TestCase):
    def setUp(self):
        self.train_data = MNISTDataProvider('train_images', 100)
        self.train_labels = MNISTDataProvider('train_labels', 100)
        self.test_data = MNISTDataProvider('test_images')
        self.test_labels = MNISTDataProvider('test_labels')
        self.D = MNISTDataProvider.D
        self.n_out = 10

    def test_relu(self):
        # import pudb; pudb.set_trace();
        model = NeuralNet(n_in=self.D, n_out=self.n_out,
                          layers=[1000], activation_function='relu',
                          dropout=True)
        optimizer = SGD(model, SimpleSGDUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=exponential_scheduler(1., .99))
        optimizer.run(20)
        self.assertLess(optimizer.progress_monitors[0].train_error[-1][1], 
                        optimizer.progress_monitors[0].train_error[0][1])
        del model, optimizer

    def test_momentum(self):
        model = NeuralNet(n_in=self.D, n_out=self.n_out,
                          layers=[1000], activation_function='relu',
                          dropout=True)
        optimizer = SGD(model, MomentumUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=exponential_scheduler(1., .99),
                        momentum_schedule=linear_scheduler_up(.5, .9, 5))
        optimizer.run(20)
        self.assertLess(optimizer.progress_monitors[0].train_error[-1][1], 
                        optimizer.progress_monitors[0].train_error[0][1])
        del model, optimizer

    def test_nesterov_momentum(self):
        model = NeuralNet(n_in=self.D, n_out=self.n_out,
                          layers=[100], activation_function='relu',
                          dropout=True)
        optimizer = SGD(model, NesterovMomentumUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=exponential_scheduler(1., .99),
                        momentum_schedule=linear_scheduler_up(.5, .9, 5))
        optimizer.run(20)
        self.assertLess(optimizer.progress_monitors[0].train_error[-1][1], 
                        optimizer.progress_monitors[0].train_error[0][1])
        del model, optimizer

class TestExtractColumns(unittest.TestCase):
    def test_extract_columns(self):
        for i in range(20):
            dtype = random.choice((np.float32, np.float64))
            N = np.random.randint(100, 1000)
            M = np.random.randint(100, 1000)
            a = np.random.randint(0, M - 1)
            b = np.random.randint(a, M)
            m = b - a
            assert m > 0

            X = curand((N, M), dtype)
            Y = extract_columns(X, a, b)

            self.assertTrue(np.all(X.get()[:,a:b] == Y.get()))

if __name__ == '__main__':
    unittest.main()
