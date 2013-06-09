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

    def report(self, epoch, train_error, test_error=None):
        raise NotImplementedError

class SimpleProgressMonitor(ProgressMonitor):
    def report(self, epoch, train_error, test_error=None):
        self.print_error(epoch, train_error, test_error)

    def print_error(self, epoch, train_error, test_error=None):
        if test_error is not None:
            print 'Epoch %d, Test error: %.5g, Train Loss: %.3f' % \
              (epoch, test_error, train_error),
        else:
            print 'Epoch %d, Train Loss: %.3f' % \
              (epoch, train_error),

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
        self.multitask = self.model.top_layer.n_tasks > 1


        ### Training data
        self.train_data = train_data
        self.train_targets = train_targets
        
        # self.train_data = train_data if isinstance(train_data, DataProvider) \
        #   else MiniBatchDataProvider(train_data, batch_size_train)

        # if isinstance(train_targets, DataProvider):
        #     self.train_targets = train_targets
        # elif train_targets is None:
        #     self.train_targets = DummyDataProvider()
        # else:
        #     self.train_targets = MiniBatchDataProvider(train_targets, batch_size_train)

        ### Test data
        self.test_data = test_data
        self.test_targets = test_targets
            
        # self.test_data = test_data if test_data is None or \
        #   isinstance(test_data, DataProvider) \
        #   else MiniBatchDataProvider(test_data, batch_size_test)

        # self.test_targets = test_targets if test_targets is None or \
        #   isinstance(test_targets, DataProvider) \
        #   else MiniBatchDataProvider(test_targets, batch_size_test)

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
                    self.parameter_updater.post_gradient_update(gradients, batch_size, learning_parameters)
                    

                # Evaluate on test data
                if self.test_data is not None:
                    test_loss = 0 if self.multitask \
                      else np.zeros((self.model.top_layer.n_tasks,))
                    for batch_idx, (batch_data, batch_targets) in \
                      enumerate(izip(self.test_data, self.test_targets)):

                        test_loss += self.model.test_error(batch_data, 
                                                                batch_targets, 
                                                                average=False)

                    test_loss_rate = test_loss / float(self.N_test)
                    
                    self.progress_monitor.report(self.epoch, train_loss, test_loss_rate)
                    self.test_error.append(test_loss_rate)

                    if early_stopping and test_loss_rate < self.best_test_loss:
                        print ' (new best)'
                        self.best_test_loss = test_loss_rate
                        self.best_params = [p.copy() for p in self.model.parameters]
                        assert self.best_params[0] is not self.model.parameters[0]
                        self.best_epoch = self.epoch
                    else:
                        print

                else:
                    print 'Epoch %d, Train loss: %.3f' % \
                      (self.epoch, train_loss)

                self.train_error.append(train_loss)

                epoch_t = time.time() - t
                self.avg_epoch_t = ((self.epoch - 1) * self.avg_epoch_t + epoch_t) / self.epoch \
                  if self.avg_epoch_t is not None else epoch_t
                  
            except KeyboardInterrupt:
                print "Keyboard interrupt. Stopping training and cleaning up."
                done_looping=True

        end_time = time.clock() 
        self.train_time = end_time - start_time / 60.

        if early_stopping and self.best_params is not None:
            self.model.parameters = self.best_params

        print "Optimization complete. Best test error of %.5g obtained in self.epoch %d" % \
          (self.best_test_loss, self.best_epoch)
        print "Runtime: %.2fm" % self.train_time
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t
        self.time = datetime.datetime.now()

    def norm_v_norm(self):
        if self.max_vec_norm:
            for w in self.model.parameters:
                if len(w.shape) == 2:
                    vector_normalize(w, self.max_vec_norm)
