from .utils.math import ceil_div
import numpy as np
import os
from hebel.optimizers import SGD

def cross_validation(config, data, save_model_path, make_model_func,
                     make_data_provider_func, make_progress_monitor_func, get_stats_func,
                     make_figures_func):

    n_folds = config['n_folds']
    n_data = config['n_data']
    validation_share = config['validation_share']
    
    fold_size = ceil_div(n_data, n_folds)
    N_train_validate = n_data - fold_size
    N_train = int(np.ceil((1. - validation_share) * N_train_validate))

    models_cv = []
    progress_monitors_cv = []
    fold_idx = []

    cv_stats = []

    train_error = {
        'training_error': [],
        'validation_error': []
    }

    predictions = None

    np.random.seed(config.get('numpy_seed'))
    open(os.path.join(save_model_path, 'config.yaml'), 'w').write(config['yaml_config'])

    for k in range(n_folds):
        fold_range = (k*fold_size, min((k+1)*fold_size, n_data))
        test_idx = np.arange(fold_range[0], fold_range[1], dtype=np.int32)

        train_validate_idx = np.random.permutation(
            np.r_[np.arange(0, fold_range[0], dtype=np.int32),
                  np.arange(fold_range[1], n_data, dtype=np.int32)])
        train_idx = train_validate_idx[:N_train]
        validate_idx = train_validate_idx[N_train:]

        fold_idx.append({
            'test_idx': test_idx,
            'train_idx': train_idx,
            'validate_idx': validate_idx
        })

        dp_train = make_data_provider_func(train_idx, data, config['batch_size'])
        dp_validate = make_data_provider_func(validate_idx, data, config['batch_size'])
        dp_test = make_data_provider_func(test_idx, data, test_idx.shape[0])

        model = make_model_func(config)
        models_cv.append(model)

        progress_monitor = make_progress_monitor_func(config, save_model_path, k)
        progress_monitors_cv.append(progress_monitor)

        learning_rate_schedule = config['learning_rate_fct'](**config['learning_rate_params'])

        momentum_schedule = config['momentum_schedule_fct'](**config['momentum_schedule_params']) \
                            if 'momentum_schedule_fct' in config else None
        
        optimizer = SGD(model, config['parameter_updater'], dp_train, dp_validate,
                        progress_monitor,
                        learning_rate_schedule=learning_rate_schedule,
                        momentum_schedule=momentum_schedule,
                        early_stopping=config.get('early_stopping', True))

        optimizer.run(config['epochs'], yaml_config=config['yaml_config'])

        stats = get_stats_func(dp_train, dp_test, model, config)
        cv_stats.append(stats)

        predictions_fold = model.feed_forward(dp_test.data).get()
        predictions = np.r_[predictions, predictions_fold] \
                      if predictions is not None else predictions_fold

        make_figures_func(model, progress_monitor, config, save_model_path, k)
        
        train_error['training_error'].append(progress_monitor.train_error)
        train_error['validation_error'].append(progress_monitor.validation_error)

    return models_cv, predictions, cv_stats, train_error
