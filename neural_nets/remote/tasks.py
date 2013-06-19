from celery import Celery, Task
from billiard import current_process
from neural_nets.config import load
import os

celery = Celery('tasks', broker='amqp://guest:guest@128.100.241.98:5672//')

class ExperimentRunner(Task):
    ignore_result = True
    abstract = True
    
    def __init__(self):
        import pycuda.driver as cuda
        cuda.init()
        self.ndevices = cuda.Device.count()
        self.devices = [cuda.Device(i) for i in range(self.ndevices)]

    @property
    def context(self):
        import pycuda.driver as cuda
        process_id = current_process().index
        cuda.init()
        device_id = process_id % self.ndevices
        context = self.devices[device_id].make_context()
        context.push()

        from scikits.cuda import linalg
        linalg.init()

        return context

@celery.task(base=ExperimentRunner)
def run_experiment(yaml_config):
    context = run_experiment.context

    config = load(yaml_config)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    run_conf['yaml_config'] = yaml_config
    run_conf['task_id'] = run_experiment.request.id
    optimizer.run(**run_conf)

    context.pop()
