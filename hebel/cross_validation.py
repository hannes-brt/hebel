from .utils.math import ceil_div
import numpy as np
import os
from hebel.optimizers import SGD

class CrossValidation(object):
    def __init__(self, config, data):

        self.n_folds = config['n_folds']
        self.n_data = config['n_data']
        self.validation_share = config['validation_share']

        self.fold_size = ceil_div(self.n_data, self.n_folds)
        self.N_train_validate = self.n_data - self.fold_size
        self.N_train = int(np.ceil((1. - self.validation_share) * self.N_train_validate))

        self.models_cv = []
        self.progress_monitors_cv = []
        self.fold_idx = []

        self.fold_stats = []

        self.train_error = {
            'training_error': [],
            'validation_error': []
        }

        self.predictions = None
        self.config = config
        self.data = data

        np.random.seed(config.get('numpy_seed'))

    def run_fold(self, k):
        fold_range = (k*self.fold_size, min((k+1)*self.fold_size, self.n_data))
        test_idx = np.arange(fold_range[0], fold_range[1], dtype=np.int32)

        train_validate_idx = np.random.permutation(
            np.r_[np.arange(0, fold_range[0], dtype=np.int32),
                  np.arange(fold_range[1], self.n_data, dtype=np.int32)])
        train_idx = train_validate_idx[:self.N_train]
        validate_idx = train_validate_idx[self.N_train:]

        self.fold_idx.append({
            'test_idx': test_idx,
            'train_idx': train_idx,
            'validate_idx': validate_idx
        })

        dp_train = self.make_data_provider(train_idx, self.config['batch_size'])
        dp_validate = self.make_data_provider(validate_idx, self.config['batch_size'])
        dp_test = self.make_data_provider(test_idx, test_idx.shape[0])

        model = self.make_model()
        self.models_cv.append(model)

        progress_monitor = self.make_progress_monitor(k)
        self.progress_monitors_cv.append(progress_monitor)

        learning_rate_schedule = self.config['learning_rate_fct'](**self.config['learning_rate_params'])

        momentum_schedule = self.config['momentum_schedule_fct'](**self.config['momentum_schedule_params']) \
                            if 'momentum_schedule_fct' in self.config else None
        
        optimizer = SGD(model, self.config['parameter_updater'], dp_train, dp_validate,
                        progress_monitor,
                        learning_rate_schedule=learning_rate_schedule,
                        momentum_schedule=momentum_schedule,
                        early_stopping=self.config.get('early_stopping', True))

        optimizer.run(self.config['epochs'], yaml_config=self.config['yaml_config'])

        stats = self.get_stats(dp_train, dp_test, model)
        self.fold_stats.append(stats)

        predictions_fold = model.feed_forward(dp_test.data).get()
        self.predictions = np.r_[self.predictions, predictions_fold] \
                           if self.predictions is not None else predictions_fold

        self.make_figures(model, progress_monitor, k)
        
        self.train_error['training_error'].append(progress_monitor.train_error)
        self.train_error['validation_error'].append(progress_monitor.validation_error)

    def run(self):
        for k in range(self.n_folds):
            self.run_fold(k)

    def make_data_provider(self, idx, batch_size):
        raise NotImplementedError

    def make_model(self):
        raise NotImplementedError

    def make_progress_monitor(self, fold):
        raise NotImplementedError

    def get_stats_func(self, dp_train, dp_test, model):
        return {}

    def make_figures(self, model, progress_monitor, fold):
        pass

    def post_run(self):
        pass