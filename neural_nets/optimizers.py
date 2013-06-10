import numpy as np
import time, datetime, operator
from .pycuda_ops import eps
from .pycuda_ops.matrix import vector_normalize
from itertools import izip
from data_providers import DataProvider, MiniBatchDataProvider
from schedulers import constant_scheduler
from pycuda import gpuarray, cumath

class ProgressMonitor(object):
    def __init__(self, model):
        self.model = model
        self.train_error = []
        self.test_error = []

    def report(self, epoch, train_error, test_error=None):
        raise NotImplementedError

class SimpleProgressMonitor(ProgressMonitor):
    def report(self, epoch, train_error, test_error=None):
        self.train_error.append((epoch, train_error))
        if test_error is not None:
            self.test_error.append(test_error)
        self.print_error(epoch, train_error, test_error)

    def print_error(self, epoch, train_error, test_error=None):
        if test_error is not None:
            print 'Epoch %d, Test error: %.5g, Train Loss: %.3f' % \
              (epoch, test_error, train_error)
        else:
            print 'Epoch %d, Train Loss: %.3f' % \
              (epoch, train_error)

    def avg_weight(self):
        print "\nAvg weights:"

        i = 0
        for param in self.model.parameters:
            if len(param.shape) != 2: continue
            param_cpu = np.abs(param.get())
            mean_weight = param_cpu.mean()
            std_weight = param_cpu.std()
            print 'Layer %d: %.4f [%.4f]' % (i, mean_weight, std_weight) 
            i += 1

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

        self.progress_monitor = SimpleProgressMonitor(self.model)
        self.early_stopping_module = EarlyStoppingModule(self.model)

    def run(self, iter=200, test_interval=5, 
            early_stopping=True):
        # Initialize variables
        self.epoch = 0
        done_looping = False
        start_time = time.clock()
        self.avg_epoch_t = None           # Average time for one epoch

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
                                        
                    self.progress_monitor.report(self.epoch, train_loss, test_loss_rate)

                else:
                    self.progress_monitor.report(self.epoch, train_loss)

                epoch_t = time.time() - t
                self.avg_epoch_t = ((self.epoch - 1) * self.avg_epoch_t + epoch_t) / self.epoch \
                  if self.avg_epoch_t is not None else epoch_t
                  
            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping=True

        end_time = time.clock() 
        self.train_time = end_time - start_time / 60.
        
        if self.early_stopping_module is not None:
            self.early_stopping_module.finish()

        print "Runtime: %.2fm" % self.train_time
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t
        self.time = datetime.datetime.now()

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)
