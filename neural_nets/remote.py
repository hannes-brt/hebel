# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from celery import Celery, Task
from billiard import current_process
from neural_nets.config import load
import os
from itertools import izip
import pycuda.driver as cuda

celery = Celery('tasks', broker='amqp://guest:guest@128.100.241.98:5672//')

class ExperimentRunner(Task):
    ignore_result = True
    abstract = True
    
    def make_context(self):
        cuda.init()
        self.ndevices = cuda.Device.count()

        process_id = current_process().index
        cuda.init()
        device_id = process_id % self.ndevices
        self.context = cuda.Device(device_id).make_context()
        self.context.push()

        from scikits.cuda import linalg
        linalg.init()

@celery.task(base=ExperimentRunner)
def run_experiment(yaml_config):
    if cuda.Context.get_current() is None:
        run_experiment.make_context()
    
    config = load(yaml_config)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    run_conf['yaml_config'] = yaml_config
    run_conf['task_id'] = run_experiment.request.id
    optimizer.run(**run_conf)

    if config.has_key('test_dataset'):
        test_data = config['test_dataset']['test_data']
        model = optimizer.model
        progress_monitor = optimizer.progress_monitor

        test_error = 0
        for batch_data, batch_targets in test_data:
            test_error += model.test_error(batch_data, batch_targets, average=False)
        test_error /= float(test_data.N)
        progress_monitor.test_error = test_error
        
