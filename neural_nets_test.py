import unittest
import numpy as np
import pycuda.autoinit
from neural_nets.models import NeuralNet
from neural_nets.optimizers import SGD
from neural_nets.parameter_updaters import SimpleSGDUpdate, MomentumUpdate, NesterovMomentumUpdate
from neural_nets.data_providers import MiniBatchDataProvider, BatchDataProvider, MNISTDataProvider
from neural_nets.schedulers import constant_scheduler, exponential_scheduler, linear_scheduler_up

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

if __name__ == '__main__':
    unittest.main()
