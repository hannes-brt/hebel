import unittest
import pycuda.autoinit
from neural_nets.models import NeuralNet
from neural_nets.optimizers import SGD
from neural_nets.parameter_updaters import SimpleSGDUpdate, MomentumUpdate, NesterovMomentumUpdate
from neural_nets.data_providers import MiniBatchDataProvider, BatchDataProvider, MNISTDataProvider
from neural_nets.schedulers import constant_scheduler, exponential_scheduler, linear_scheduler_up

class TestNeuralNetMNIST(unittest.TestCase):
    def setUp(self):
        self.train_data = MNISTDataProvider(100, 'train_images')
        self.train_labels = MNISTDataProvider(100, 'train_labels')
        self.test_data = MNISTDataProvider(None, 'test_images')
        self.test_labels = MNISTDataProvider(None, 'test_labels')
        self.D = MNISTDataProvider.D
        self.n_out = 10

    def test_relu(self):
        model = NeuralNet(self.D, self.n_out, [100], 'relu', dropout=True)
        optimizer = SGD(model, SimpleSGDUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=constant_scheduler(1.5))
        optimizer.run(10)
        self.assertLess(optimizer.train_error[-1], optimizer.train_error[0])

    def test_momentum(self):
        model = NeuralNet(self.D, self.n_out, [100], 'relu', dropout=True)
        optimizer = SGD(model, MomentumUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=constant_scheduler(1.5),
                        momentum_schedule=linear_scheduler_up(.5, .9, 5))
        optimizer.run(10)
        self.assertLess(optimizer.train_error[-1], optimizer.train_error[0])

    def test_nesterov_momentum(self):
        model = NeuralNet(self.D, self.n_out, [100], 'relu', dropout=True)
        optimizer = SGD(model, NesterovMomentumUpdate, self.train_data,
                        self.train_labels, self.test_data, self.test_labels,
                        learning_rate_schedule=constant_scheduler(1.5),
                        momentum_schedule=linear_scheduler_up(.5, .9, 5))
        optimizer.run(10)
        self.assertLess(optimizer.train_error[-1], optimizer.train_error[0])

if __name__ == '__main__':
    unittest.main()
