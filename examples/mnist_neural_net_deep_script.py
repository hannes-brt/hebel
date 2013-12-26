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
                  layers=[2000, 2000, 2000, 500],
                  activation_function='relu',
                  dropout=True, input_dropout=0.2)

# Create optimizer object
progress_monitor = ProgressMonitor(
    experiment_name='mnist',
    save_model_path='examples/mnist',
    save_interval=5,
    output_to_log=True)

optimizer = SGD(model, MomentumUpdate, train_data, validation_data,
                learning_rate_schedule=exponential_scheduler(5., .995),
                momentum_schedule=linear_scheduler_up(.1, .9, 100))

# Run model
optimizer.run(500)

# Evaulate error on test set
test_error = model.test_error(test_data)
print "Error on test set: %.3f" % test_error
