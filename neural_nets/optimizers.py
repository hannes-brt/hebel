import numpy as np
import time, cPickle, os, inspect
from datetime import datetime
from .pycuda_ops import eps
from .pycuda_ops.matrix import vector_normalize
from itertools import izip
from .data_providers import DataProvider, MiniBatchDataProvider
from .schedulers import constant_scheduler
from .monitors import SimpleProgressMonitor
from pycuda import gpuarray, cumath

class EarlyStoppingModule(object):
    def __init__(self, model):
        self.model = model
        self.best_test_loss = np.inf

    def update(self, epoch, test_loss):
        if test_loss < self.best_test_loss:
            print '* ',
            self.best_test_loss = test_loss
            self.best_params = [p.copy() for p in self.model.parameters]
            assert self.best_params[0] is not self.model.parameters[0]
            self.best_epoch = epoch

    def finish(self):
        self.model.parameters = self.best_params
        print "Optimization complete. Best test error of %.5g obtained in self.epoch %d" % \
          (self.best_test_loss, self.best_epoch)
        

class SGD(object):
    def __init__(self,
                 model, parameter_updater, 
                 train_data, train_targets, 
                 test_data=None, test_targets=None,
                 progress_monitors=None,
                 learning_rate_schedule=constant_scheduler(.1),
                 momentum_schedule=None,
                 batch_size_train=100, batch_size_test=100):

        """ Stochastic gradient descent
        """

        ### Initialization
        
        self.model = model

        ### Training data
        self.train_data = train_data
        self.train_targets = train_targets
        
        ### Test data
        self.test_data = test_data
        self.test_targets = test_targets
            
        ### Data size
          
        self.N_train = self.train_data.N

        if test_data is not None:
            self.N_test = self.test_data.N

        ### Learning rate schedule
        self.learning_parameter_iterators = [learning_rate_schedule]

        ### Momentum, rmsprop, etc

        self.parameter_updater = parameter_updater(self.model)
        
        if momentum_schedule is not None:
            self.learning_parameter_iterators.append(momentum_schedule)

        if progress_monitors is None:
            self.progress_monitors = [SimpleProgressMonitor(self.model)]
        else:
            self.progress_monitors = progress_monitors

        for i in range(len(self.progress_monitors)):
            if inspect.isclass(self.progress_monitors[i]):
                self.progress_monitors[i] = self.progress_monitors[i]()
            if self.progress_monitors[i].model is None:
                self.progress_monitors[i].model = model
            
        self.early_stopping_module = EarlyStoppingModule(self.model)

    def run(self, iter=200, test_interval=5,
            early_stopping=True, yaml_config=None):
        # Initialize variables
        self.epoch = 0
        done_looping = False
        
        map(lambda pm: pm.start_training(), self.progress_monitors)

        for pm in self.progress_monitors:
            pm.yaml_config = yaml_config

        # Main loop
        for self.epoch in range(self.epoch, self.epoch + iter):
            learning_parameters = map(lambda lp: lp.next(),
                                      self.learning_parameter_iterators)
            if done_looping: break
            
            try: 
                t = time.time()

                # Train on mini-batches
                train_loss = 0.
                
                for batch_idx, (batch_data, batch_targets) in \
                  enumerate(izip(self.train_data, self.train_targets)):
                    batch_size = batch_data.shape[0]
                  
                    self.parameter_updater.pre_gradient_update()

                    batch_loss, gradients = self.model.training_pass(batch_data, batch_targets)
                    train_loss += batch_loss
                    self.parameter_updater\
                      .post_gradient_update(gradients, batch_size, learning_parameters)

                # Evaluate on test data
                if self.test_data is not None and not self.epoch % test_interval:
                    test_loss = 0.
                    for batch_idx, (batch_data, batch_targets) in \
                      enumerate(izip(self.test_data, self.test_targets)):

                        test_loss += self.model.test_error(batch_data, 
                                                           batch_targets, 
                                                           average=False)

                    test_loss_rate = test_loss / float(self.N_test)

                    if self.early_stopping_module is not None:
                        self.early_stopping_module.update(self.epoch, test_loss_rate)

                    epoch_t = time.time() - t

                    map(lambda pm: pm.report(self.epoch, train_loss, test_loss_rate,
                        epoch_t=epoch_t), self.progress_monitors)
                else:
                    epoch_t = time.time() - t                    
                    map(lambda pm: pm.report(self.epoch, train_loss, epoch_t=epoch_t),
                        self.progress_monitors)

            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping=True

        if self.early_stopping_module is not None:
            self.early_stopping_module.finish()

        map(lambda pm: pm.finish_training(), self.progress_monitors)

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)
