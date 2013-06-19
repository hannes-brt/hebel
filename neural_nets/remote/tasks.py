from celery import Celery, Task
from billiard import current_process
from neural_nets.config import load
import os
from itertools import izip

celery = Celery('tasks', broker='amqp://guest:guest@128.100.241.98:5672//')

class ExperimentRunner(Task):
    ignore_result = True
    abstract = True
    
    def __init__(self):
        import pycuda.driver as cuda
        cuda.init()
        self.ndevices = cuda.Device.count()

        process_id = current_process().index
        cuda.init()
        device_id = process_id % self.ndevices
        self.context = cuda.Device(device_id).make_context()
        context.push()

        from scikits.cuda import linalg
        linalg.init()

@celery.task(base=ExperimentRunner)
def run_experiment(yaml_config):
    context = run_experiment.context

    config = load(yaml_config)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    run_conf['yaml_config'] = yaml_config
    run_conf['task_id'] = run_experiment.request.id
    optimizer.run(**run_conf)

    if config.has_key('test_dataset'):
        test_data = config['test_dataset']['test_data']
        test_targets = config['test_dataset']['test_targets']
        model = optimizer.model
        progress_monitor = optimizer.progress_monitor

        test_error = 0
        for batch_data, batch_targets in izip(test_data, test_targets):
            test_error += model.test_error(batch_data, batch_targets, average=False)
        test_error /= float(test_data.N)
        progress_monitor.test_error = test_error
        
