import numpy as np
import time, cPickle, os, sys
from datetime import datetime
from mongoengine import DoesNotExist
from .schema import Dataset, Experiment, ErrorLog

class ProgressMonitor(object):
    def __init__(self, experiment_name=None, save_model_path=None, 
                 save_interval=0, output_to_log=False, model=None,
                 dataset=None, task_id=None):

        self.experiment = Experiment()

        self.experiment_name = experiment_name
        self.save_model_path = save_model_path
        self.save_interval = save_interval
        self.output_to_log = output_to_log
        self.model = model

        try: 
            self.dataset = Dataset.objects(name=dataset).get()
        except DoesNotExist:
            self.dataset = Dataset(name=dataset)
            self.dataset.save()
            
        self.task_id = task_id
        self.train_error = []
        self.validation_error = []
        self.avg_epoch_t = None
        self._time = datetime.isoformat(datetime.now())

        self.experiment.name = self.experiment_name
        self.experiment.epochs = 0
        self.experiment.dataset = self.dataset

        self.experiment.train_error = ErrorLog()
        self.experiment.validation_error = ErrorLog()

        if self.experiment.task_id is not None:
            self.experiment.save()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

        if self._model is not None:
            self.experiment.model_checksum = model.checksum            

    @property
    def yaml_config(self):
        return self._yaml_config

    @yaml_config.setter
    def yaml_config(self, yaml_config):
        self._yaml_config = yaml_config
        yaml_path = os.path.join(self.save_path, 'yaml_config.yml')
        f = open(yaml_path, 'w')
        f.write(self._yaml_config)
        self.experiment.yaml_config = yaml_config        

    @property
    def task_id(self):
        return self.experiment.task_id

    @task_id.setter
    def task_id(self, task_id):
        if task_id is not None:
            self.experiment.task_id = task_id
            self.experiment.save()
            self.makedir()
            open(os.path.join(self.save_path, 'task_id'), 'w')\
              .write(self.experiment.task_id)

    @property
    def test_error(self):
        return self.experiment.test_error

    @test_error.setter
    def test_error(self, test_error):
        self.experiment.test_error = test_error
        if self.experiment.task_id is not None:
            self.experiment.save()

    def makedir(self):
        experiment_dir_name = '_'.join((
            self.experiment_name,
            datetime.isoformat(datetime.now()),
            self.task_id))

        path = os.path.join(self.save_model_path, 
                            experiment_dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_path = path

        if self.output_to_log:
            self.log = open(os.path.join(self.save_path, 'output.log'), 'w', 1)
            sys.stdout = self.log
            sys.stderr = self.log

    def start_training(self):
        self.start_time = datetime.now()

    def report(self, epoch, train_error, validation_error=None, epoch_t=None):
        # Print logs
        self.train_error.append((epoch, train_error))
        if validation_error is not None:
            self.validation_error.append(validation_error)
        self.print_error(epoch, train_error, validation_error)

        if epoch_t is not None:
            self.avg_epoch_t = ((epoch - 1) * \
                                self.avg_epoch_t + epoch_t) / epoch \
                                if self.avg_epoch_t is not None else epoch_t

        # Pickle model
        if not epoch % self.save_interval:
            filename = 'model_%s_epoch%04d.pkl' % (
              self.experiment_name,
              epoch)
            path = os.path.join(self.save_path, filename)
            cPickle.dump(self.model, open(path, 'wb'))

        # Save to database
        self.experiment.epochs = epoch
        self.experiment.train_error.epoch.append(epoch)
        self.experiment.train_error.value.append(train_error)

        if validation_error is not None:
            self.experiment.validation_error.epoch.append(epoch)
            self.experiment.validation_error.value.append(validation_error)

        if self.experiment.task_id is not None:
            self.experiment.save()

    def print_error(self, epoch, train_error, validation_error=None):
        if validation_error is not None:
            print 'Epoch %d, Validation error: %.5g, Train Loss: %.3f' % \
              (epoch, validation_error, train_error)
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

    def finish_training(self):
        # Print logs
        end_time = datetime.now()
        self.train_time = end_time - self.start_time
        print "Runtime: %dm %ds" % (self.train_time.total_seconds() // 60, 
                                    self.train_time.total_seconds() % 60)
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t

        # Pickle model
        filename = 'model_%s_end_%s.pkl' % (
            self.experiment_name,
            self.model.checksum)
        path = os.path.join(self.save_path, filename)
        print "Saving model to %s" % path
        cPickle.dump(self.model, open(path, 'wb'))

        if self.output_to_log:
            self.log.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # Save to database
        self.experiment.runtime = self.train_time
        self.experiment.date_finished = end_time

        if self.experiment.task_id is not None:
            self.experiment.save()

class SimpleProgressMonitor(object):
    def __init__(self):
        self.model = None

        self.train_error = []
        self.validation_error = []
        self.avg_epoch_t = None
        self._time = datetime.isoformat(datetime.now())

    def start_training(self):
        self.start_time = datetime.now()

    def report(self, epoch, train_error, validation_error=None, epoch_t=None):
        # Print logs
        self.print_error(epoch, train_error, validation_error)

        if epoch_t is not None:
            self.avg_epoch_t = ((epoch - 1) * \
                                self.avg_epoch_t + epoch_t) / epoch \
                                if self.avg_epoch_t is not None else epoch_t

    def print_error(self, epoch, train_error, validation_error=None):
        if validation_error is not None:
            print 'Epoch %d, Validation error: %.5g, Train Loss: %.3f' % \
              (epoch, validation_error, train_error)
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

    def finish_training(self):
        # Print logs
        end_time = datetime.now()
        self.train_time = end_time - self.start_time
        print "Runtime: %dm %ds" % (self.train_time.total_seconds() // 60, 
                                    self.train_time.total_seconds() % 60)
        print "Avg. time per epoch %.2fs" % self.avg_epoch_t
