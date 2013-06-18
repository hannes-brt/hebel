from celery import Celery, Task
from neural_nets.config import load
import os

celery = Celery('tasks', broker='amqp://guest:guest@localhost:5673//')

class ExperimentRunner(Task):
    ignore_result = True
    abstract = True
    
    def __init__(self):
        import pycuda.driver as cuda
        cuda.init()
        self.ndevices = cuda.Device.count()
        self.devices = [cuda.Device(i) for i in range(self.ndevices)]
        self.task_count = 0

    @property
    def context(self):
        import pycuda.driver as cuda
        self.task_count += 1
        cuda.init()
        device_id = os.getpid() % self.ndevices
        context = self.devices[device_id].make_context()
        context.push()
        print context.get_device().pci_bus_id()

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
        optimizer.run(**run_conf)
    
        context.pop()
