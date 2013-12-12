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

"""Example of neural net with a linear regression output layer, using
the Boston dataset.

"""

def main():
    import numpy as np
    import pycuda.autoinit
    from pycuda import gpuarray
    from skdata import toy
    from hebel.data_providers import BatchDataProvider
    from hebel.models import NeuralNetRegression
    from hebel.optimizers import SGD
    from hebel.parameter_updaters import SimpleSGDUpdate
    from hebel.monitors import SimpleProgressMonitor
    from hebel.schedulers import exponential_scheduler

    # Get data
    data_cpu, targets_cpu = toy.Boston().regression_task()
    data = gpuarray.to_gpu(data_cpu.astype(np.float32))
    targets = gpuarray.to_gpu(targets_cpu.astype(np.float32))
    data_provider = BatchDataProvider(data, targets)

    # Create model object
    model = NeuralNetRegression(n_in=data_cpu.shape[1], n_out=targets_cpu.shape[1],
                                layers=[100], activation_function='relu')
    
    # Create optimizer object
    optimizer = SGD(model, SimpleSGDUpdate, data_provider, data_provider,
                    learning_rate_schedule=exponential_scheduler(.1, .9999),
                    early_stopping=True)
    optimizer.run(3000)
    
if __name__ == "__main__":
    main()
