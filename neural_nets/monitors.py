import numpy as np
import time, cPickle, os
from datetime import datetime

class ProgressMonitor(object):
    def __init__(self, model=None):
        self.model = model
        self.train_error = []
        self.test_error = []
        self.avg_epoch_t = None

    def start_training(self):
        raise NotImplementedError

    def finish_training(self):
        raise NotImplementedError

    def report(self, epoch, train_error, test_error=None):
        raise NotImplementedError

class SimpleProgressMonitor(ProgressMonitor):
    def report(self, epoch, train_error, test_error=None, epoch_t=None):
        self.train_error.append((epoch, train_error))
        if test_error is not None:
            self.test_error.append(test_error)
        self.print_error(epoch, train_error, test_error)

        if epoch_t is not None:
            self.avg_epoch_t = ((epoch - 1) * \
                                self.avg_epoch_t + epoch_t) / epoch \
                                if self.avg_epoch_t is not None else epoch_t

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

    def start_training(self):
        self.start_time = datetime.now()

    def finish_training(self, save_model_path=None):
        end_time = datetime.now()
        self.train_time = (end_time - self.start_time).total_seconds()
        print "Runtime: %dm %ds" % (self.train_time // 60, self.train_time % 60)
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t

        if save_model_path is not None:
            print "Saving model to %s" % save_model_path
            cPickle.dump(self.model, open(save_model_path, 'wb'))

class ModelSaver(ProgressMonitor):    
    def __init__(self, experiment_name, save_model_path, save_interval, model=None):
        super(ModelSaver, self).__init__()
        self.experiment_name = experiment_name
        self.save_model_path = save_model_path
        self.save_interval = save_interval
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

        if self._model is not None:
            self.makedir()

    @property
    def yaml_config(self):
        return self._yaml_config

    @yaml_config.setter
    def yaml_config(self, yaml_config):
        self._yaml_config = yaml_config
        yaml_path = os.path.join(self.save_path, 'yaml_config.yml')
        f = open(yaml_path, 'w')
        f.write(self._yaml_config)

    def makedir(self):
        experiment_dir_name = '_'.join((
            self.experiment_name,
            datetime.isoformat(datetime.now()),
            self.model.checksum))

        path = os.path.join(self.save_model_path, 
                            experiment_dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_path = path

    def report(self, epoch, train_error, test_error=None, epoch_t=None):
        if not epoch % self.save_interval:
            filename = 'model_%s_epoch%04d_%s.pkl' % (
              self.experiment_name,
              epoch,
              self.model.checksum)
            path = os.path.join(self.save_path, filename)
            cPickle.dump(self.model, open(path, 'wb'))

    def start_training(self):
        pass

    def finish_training(self):
        filename = 'model_%s_end_%s.pkl' % (
            self.experiment_name,
            self.model.checksum)
        path = os.path.join(self.save_path, filename)
        cPickle.dump(self.model, open(path, 'wb'))

