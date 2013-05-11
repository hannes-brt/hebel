import numpy as np
import time
import datetime
from .pycuda_ops import eps, vector_normalize
from itertools import izip
from data_providers import DataProvider, MiniBatchDataProvider
from schedulers import constant_scheduler
from pycuda import gpuarray, cumath

class SGD(object):
    def __init__(self,
                 model, train_data, train_targets, test_data, test_targets,
                 learning_rate_schedule,
                 momentum_schedule=None,
                 nesterov_momentum=False,
                 rmsprop=False,
                 max_vec_norm=False,
                 epsilon=1e-6,
                 batch_size_train=100, batch_size_test=100,
                 introspection_prefix=None):

        """ Stochastic gradient descent
        """

        # self.batch_size_train = batch_size_train
        # self.batch_size_test = batch_size_test

        self.model = model

        # if self.batch_size_train is None: self.batch_size_train = self.N_train
        # if self.batch_size_test is None: self.batch_size_test = self.N_test

        self.train_data = train_data if isinstance(train_data, DataProvider) \
          else MiniBatchDataProvider(train_data, batch_size_train)

        if isinstance(train_targets, DataProvider):
            self.train_targets = train_targets
        elif train_targets is None:
            self.train_targets = DummyDataProvider()
        else:
            self.train_targets = MiniBatchDataProvider(train_targets, batch_size_train)

        self.test_data = test_data if isinstance(test_data, DataProvider) \
          else MiniBatchDataProvider(test_data, batch_size_test)

        if isinstance(test_targets, DataProvider):
            self.test_targets = test_targets
        elif test_targets is None:
            self.test_targets = DummyDataProvider()
        else:
            self.test_targets = MiniBatchDataProvider(test_targets, batch_size_test)

        self.N_train = self.train_data.N
        self.N_test = self.test_data.N

        self.learning_rate_schedule = learning_rate_schedule

        if momentum_schedule:
            self.momentum_schedule = momentum_schedule
        else:
            self.momentum_schedule = constant_scheduler(0.)
        
        self.nesterov_momentum = nesterov_momentum
        self.rmsprop = rmsprop
        self.max_vec_norm = max_vec_norm
        self.epsilon = epsilon
        self.introspection_prefix = introspection_prefix

        # self.n_train_batches = self.train_data.shape[0] // self.batch_size_train
        # self.n_test_batches = self.test_data.shape[0] // self.batch_size_test

        if self.introspection_prefix is not None:
            from introspection.client import IntrospectionClient
            self.radio = IntrospectionClient(['test_loss', 'epoch'], 
                                             prefix=self.introspection_prefix)

        self.test_error = []
        self.train_error = []

    def run(self, iter=200, early_stopping=True):
        # Initialize variables
        self.epoch = 0
        done_looping = False
        self.best_params = None
        self.best_test_loss = np.inf
        prev_epoch_loss = np.inf
        self.best_epoch = 0
        start_time = time.clock()
        self.velocity = None
        self.rmsprop_avg = len(self.model.parameters) * [1.]

        self.avg_epoch_t = None           # Average time for one epoch


        # Main loop
        for self.epoch, self.momentum, self.learning_rate in \
          zip(range(self.epoch, self.epoch + iter),
              self.momentum_schedule, 
              self.learning_rate_schedule):
            
            if done_looping: break
            
            try: 
                t = time.time()

                # Train on mini-batches
                train_loss = 0
                
                for batch_idx, (batch_data, batch_targets) in \
                  enumerate(izip(self.train_data, self.train_targets)):

                    if self.nesterov_momentum:
                        self.nesterov_momentum_move()

                    batch_loss, gradients = self.model.training_pass(batch_data, batch_targets)
                    train_loss += batch_loss

                    self.norm_v_norm()

                    # if self.rmsprop:
                    #     gradients = self.rmsprop_update(gradients)

                    if self.nesterov_momentum:
                          self.nesterov_gradient_move(gradients)
                    elif self.momentum:
                          self.momentum_update(gradients)
                    else:
                        self.simple_sgd_update(gradients)

                    del gradients
                    del batch_data
                    del batch_targets

                # Evaluate on test data
                test_loss = 0
                for batch_idx, (batch_data, batch_targets) in \
                  enumerate(izip(self.test_data, self.test_targets)):

                    test_loss += self.model.test_error(batch_data, batch_targets, average=False)
                    
                test_loss_rate = test_loss / float(self.N_test)

                print 'Epoch %d, Test error: %.5g, Train Loss: %.3f' % \
                  (self.epoch, test_loss_rate, train_loss),

                self.test_error.append(test_loss_rate)
                self.train_error.append(train_loss)

                if self.introspection_prefix is not None:
                    self.radio.send(epoch=self.epoch, test_loss=test_loss_rate)

                if test_loss_rate < self.best_test_loss:
                    print ' (new best)'
                    self.best_test_loss = test_loss_rate
                    self.best_params = map(lambda param: param.copy(), self.model.parameters)
                    # self.best_params = self.model.parameters
                    self.best_epoch = self.epoch
                else:
                    print

                epoch_t = time.time() - t
                self.avg_epoch_t = ((self.epoch - 1) * self.avg_epoch_t + epoch_t) / self.epoch \
                  if self.avg_epoch_t is not None else epoch_t
                  
            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping=True                

        end_time = time.clock() 
        self.train_time = end_time - start_time / 60.

        if early_stopping:
            self.model.parameters = self.best_params

        print "Optimization complete. Best test error of %.5g obtained in self.epoch %d" % \
          (self.best_test_loss, self.best_epoch)
        print "Runtime: %.2fm" % self.train_time
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t
        self.time = datetime.datetime.now()

    def simple_sgd_update(self, gradients):
        new_parameters = []
        for param, gparam, lr_multiplier \
          in zip(self.model.parameters, gradients, self.model.lr_multiplier):
            param -= self.learning_rate * gparam * lr_multiplier
            new_parameters.append(param)
        self.model.parameters = new_parameters

    def momentum_update(self, gradients):
        new_velocity = []

        if self.velocity is None: 
            self.velocity = len(self.model.parameters) * [None]

        new_parameters = []
        for param, gparam, vparam, lr_multiplier in \
          zip(self.model.parameters, gradients, self.velocity, 
              self.model.lr_multiplier):

            if vparam is not None and self.momentum > 0.:
                update = self.momentum * vparam \
                  - self.learning_rate * gparam * lr_multiplier
            else:
                update = - self.learning_rate * gparam * lr_multiplier

            param += update
            new_parameters.append(param)
            new_velocity.append(update)

        self.model.parameters = new_parameters
        if self.momentum: self.velocity = new_velocity

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)

    def nesterov_momentum_move(self):
        """ First step of Nesterov momentum method:
        take step in direction of accumulated gradient
        """

        if self.velocity is None: 
            self.velocity = len(self.model.parameters) * [None]

        new_parameters = []
        for (param, vparam) in zip(self.model.parameters, self.velocity):
            if vparam is not None:
                param += self.momentum * vparam
            new_parameters.append(param)

        self.model.parameters = new_parameters

    def nesterov_gradient_move(self, gradients):
        """ Second step of Nesterov momentum method:
        take step in direction of new gradient and update velocity
        """

        new_velocity = []
        if self.velocity is None: self.velocity = len(self.model.parameters) * [None]

        new_parameters = []
        for param, gparam, vparam, lr_multiplier in \
          zip(self.model.parameters, gradients, 
              self.velocity, self.model.lr_multiplier):
            update = - self.learning_rate * gparam * lr_multiplier
            if vparam is not None:
                new_velocity.append(self.momentum * vparam + update)
            else:
                new_velocity.append(update)            
            param += update
            new_parameters.append(param)

        self.model.parameters = new_parameters
        self.velocity = new_velocity

    # def rmsprop_update(self, gradient):
    #     new_rmsprop_avg = []
    #     new_gradient = []
        
    #     for r, g in zip(self.rmsprop_avg, gradient):
    #         new_r = self.rmsprop * r + (1 - self.rmsprop) * g**2. + eps

    #         if gpu.is_garray(g):
    #             new_g = g / gpu.sqrt(new_r)
    #         else:
    #             new_g = g / np.sqrt(new_r)
    #         new_gradient.append(new_g)
    #         new_rmsprop_avg.append(new_r)

    #     self.rmsprop_avg = new_rmsprop_avg
    #     return new_gradient
