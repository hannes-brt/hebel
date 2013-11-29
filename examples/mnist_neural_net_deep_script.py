import pycuda.autoinit
from hebel.models import NeuralNet
from hebel.optimizers import SGD
from hebel.parameter_updaters import MomentumUpdate
from hebel.data_providers import MNISTDataProvider
from hebel.monitors import ProgressMonitor
from hebel.schedulers import exponential_scheduler, linear_scheduler_up

# Initialize data providers
train_data = MNISTDataProvider('train', batch_size=100)
validation_data = MNISTDataProvider('val')
test_data = MNISTDataProvider('test')

D = train_data.D                        # Dimensionality of inputs 
K = 10                                  # Number of classes

# Create model object
model = NeuralNet(n_in=train_data.D, n_out=K,
                  layers=[1000, 500, 500],
                  activation_function='relu',
                  dropout=True)

# Create optimizer object
optimizer = SGD(model, MomentumUpdate, train_data, validation_data,
                learning_rate_schedule=exponential_scheduler(1., .995),
                momentum_schedule=linear_scheduler_up(.5, .9, 10))

# Run model
optimizer.run(100)

# Evaulate error on test set
test_error = 0
for batch_data, batch_targets in test_data:
    test_error += model.test_error(batch_data, batch_targets, average=False)
test_error /= float(test_data.N)
print "Error on test set: %.3f" % test_error
