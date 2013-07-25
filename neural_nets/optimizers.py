import numpy as np
import time, cPickle, os, inspect
from datetime import datetime
from .pycuda_ops import eps
from .pycuda_ops.matrix import vector_normalize
from itertools import izip
from .data_providers import DataProvider, MiniBatchDataProvider
from .schedulers import constant_scheduler
from .monitors import ProgressMonitor
from pycuda import gpuarray, cumath

class EarlyStoppingModule(object):
    def __init__(self, model):
        self.model = model
        self.best_validation_loss = np.inf

    def update(self, epoch, validation_loss):
        if validation_loss < self.best_validation_loss:
            print '* ',
            self.best_validation_loss = validation_loss
            self.best_params = [p.copy() for p in self.model.parameters]
            assert self.best_params[0] is not self.model.parameters[0]
            self.best_epoch = epoch

    def finish(self):
        self.model.parameters = self.best_params
        print "Optimization complete. Best validation error of %.5g obtained in self.epoch %d" % \
          (self.best_validation_loss, self.best_epoch)
        

class SGD(object):
    @property
    def best_validation_loss(self):
        return self.early_stopping_module.best_validation_loss
    
    def __init__(self,
                 model, parameter_updater, 
                 train_data, 
                 validation_data=None,
                 progress_monitor=None,
                 learning_rate_schedule=constant_scheduler(.1),
                 momentum_schedule=None,
                 batch_size_train=100, batch_size_validation=100):

        """ Stochastic gradient descent
        """

        ### Initialization
        
        self.model = model

        ### Training data
        self.train_data = train_data
        
        ### Validation data
        self.validation_data = validation_data
            
        ### Data size
        self.N_train = self.train_data.N

        if validation_data is not None:
            self.N_validation = self.validation_data.N

        ### Learning rate schedule
        self.learning_parameter_iterators = [learning_rate_schedule]

        ### Momentum, rmsprop, etc

        self.parameter_updater = parameter_updater(self.model)
        
        if momentum_schedule is not None:
            self.learning_parameter_iterators.append(momentum_schedule)

        if progress_monitor is None:
            self.progress_monitor = ProgressMonitor(model=self.model)
        else:
            self.progress_monitor = progress_monitor

        if self.progress_monitor.model is None:
            self.progress_monitor.model = self.model

        self.early_stopping_module = EarlyStoppingModule(self.model)

    def run(self, iter=200, validation_interval=5,
            early_stopping=True, yaml_config=None,
            task_id=None):
        # Initialize variables
        self.epoch = 0
        done_looping = False
        
        self.progress_monitor.start_training()

        self.progress_monitor.task_id = task_id
        self.progress_monitor.yaml_config = yaml_config

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
                  enumerate(self.train_data):
                    batch_size = batch_data.shape[0]
                  
                    self.parameter_updater.pre_gradient_update()

                    batch_loss, gradients = self.model.training_pass(batch_data, batch_targets)
                    train_loss += batch_loss
                    self.parameter_updater\
                      .post_gradient_update(gradients, batch_size, learning_parameters)

                # Evaluate on validation data
                if self.validation_data is not None and not self.epoch % validation_interval:
                    validation_loss = 0.
                    for batch_idx, (batch_data, batch_targets) in \
                      enumerate(self.validation_data):

                        validation_loss += self.model.test_error(batch_data, 
                                                                 batch_targets, 
                                                                 average=False)

                    validation_loss_rate = validation_loss / float(self.N_validation)

                    if self.early_stopping_module is not None:
                        self.early_stopping_module.update(self.epoch, validation_loss_rate)

                    epoch_t = time.time() - t

                    self.progress_monitor.report(self.epoch, train_loss, validation_loss_rate,
                                                 epoch_t=epoch_t)
                else:
                    epoch_t = time.time() - t                    
                    self.progress_monitor.report(self.epoch, train_loss, epoch_t=epoch_t)

            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping=True

        if self.early_stopping_module is not None:
            self.early_stopping_module.finish()

        self.progress_monitor.finish_training()

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)
