Getting Started
***************

There are two basic methods how you can run Hebel:

#. You can write a YAML configuration file that describes your model
   architecture, data set, and hyperparameters and run it using the
   :file:`train_model.py` script.
#. In your own Python script or program, you can create instances of
   models and optimizers programmatically.

The first makes estimating a model the easiest, as you don't have to
write any actual code. You simply specify all your parameters and data
set in an easy to read YAML configuration file and pass it to the
:file:`train_model.py` script. The script will create a directory for your
results where it will save intermediary models (in pickle-format), the
logs and final results.

The second method gives you more control over how exactly the model is
estimated and lets you interact with Hebel from other Python programs.


Running models from YAML configuration files
============================================

If you check the example YAML files in ``examples/`` you will see that the configuration file defines three top-level sections:

#. ``run_conf``: These options are passed to the method
   :meth:`hebel.optimizers.SGD.run()`.
#. ``optimizer``: Here you instantiate a :class:`hebel.optimizers.SGD`
   object, including the model you want to train and the data to use
   for training and validation.
#. ``test_dataset``: This section is optional, but here you can define
   test data to evaluate the model on after training.

Check out :file:`examples/mnist_neural_net_shallow.yml`, which
includes everything to train a one layer neural network on the `MNIST
dataset <http://yann.lecun.com/exdb/mnist/>`_:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml

You can see that the only option we pass to ``run_conf`` is the number
of iterations to train the model. 

The ``optimizer`` section is more interesting. Hebel uses the special
``!obj``, ``!import``, and ``!pkl`` directives from `PyLearn 2
<http://deeplearning.net/software/pylearn2/yaml_tutorial/index.html#yaml-tutorial>`_. The
``!obj`` directive is used most extensively and can be used to
instantiate any Python class. First the optimizer
:class:`hebel.optimizers.SGD` is instantiated and in the lines below
we are instantiating the model:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 3-17

We are designing a model with one hidden layer that has 784 input
units (the dimensionality of the MNIST data) and 2000 hidden units. We
are also using `dropout <http://arxiv.org/abs/1207.0580>`_ for
regularization. The logistic output layer uses 10 classes (the number
of classes in the MNIST data). You can also add different amounts of
L1 or L2 penalization to each layer, which we are not doing here.

.. _parameter-updaters:

Next, we define a ``parameter_updater``, which is a rule that defines
how the weights are updated given the gradients:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 18

There are currently three choices:

* :class:`hebel.parameter_updaters.SimpleSGDUpdate`, which performs
   regular gradient descent
* :class:`hebel.parameter_updaters.MomentumUpdate`, which performs
   gradient descent with momentum, and
* :class:`hebel.parameter_updaters.NesterovMomentumUpdate`, which performs
   gradient descent with Nesterov momentum.

The next two sections define the data for the model. All data must be
given as instances of ``DataProvider`` objects:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 19-25

A ``DataProvider`` is a class that defines an iterator which returns
successive minibatches of the data as well as saves some metadata,
such as the number of data points. There is a special
:class:`hebel.data_providers.MNISTDataProvider` especially for the
MNIST data. We use the standard splits for training and validation
data here. There are several ``DataProviders`` defined in
:mod:`hebel.data_providers`.

The next few lines define how some of the hyperparameters are changed
over the course of the training:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 26-31

The module :mod:`hebel.schedulers` defines several schedulers, which
are basically just simple rules how certain parameters should
evolve. Here, we define that the learning rate should decay
exponentially with a factor of 0.995 in every epoch and the momentum
should increase from 0.5 to 0.9 during the first 10 epochs and then
stay at this value.

The last entry argument to :class:`hebel.optimizers.SGD` is
``progress_monitor``:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 32-38

A progress monitor is an object that takes care of reporting periodic
progress of our model, saving snapshots of the model at regular
intervals, etc. When you are using the YAML configuration system,
you'll probably want to use :class:`hebel.monitors.ProgressMonitor`,
which will save logs, outputs, and snapshots to disk. In contrast,
:class:`hebel.monitors.SimpleProgressMonitor` will only print progress
to the terminal without saving the model itself.

Finally, you can define a test data set to be evaluated after the training completes:

.. literalinclude:: ../examples/mnist_neural_net_shallow.yml
   :lines: 40-43

Here, we are specifying the MNIST test split.

Once you have your configuration file defined, you can run it such as in::

  python train_model.py examples/mnist_neural_net_shallow.yml

The script will create the output directory you specified in
``save_model_path`` if it doesn't exist yet and start writing the log
into a file called ``output_log``. If you are interested in keeping an
eye on the training process you can check on that file with::

  tail -f output_log

Using Hebel in Your Own Code
============================

If you want more control over the training procedure or integrate
Hebel with your own code, then you can use Hebel programmatically. 

For an example, have a look at :file:`examples/mnist_neural_net_deep_script.py`:

.. literalinclude:: ../examples/mnist_neural_net_deep_script.py

There are three basic tasks you have to do to train a model in Hebel:

#. Define the data you want to use for training, validation, or
   testing using ``DataProvider`` objects,
#. instantiate a ``Model`` object, and
#. instantiate an ``SGD`` object that will train the model using
   stochastic gradient descent.

Defining a Data Set
-------------------

In this example we're using the MNIST data set again through the
:class:`hebel.data_providers.MNISTDataProvider` class:

.. literalinclude:: ../examples/mnist_neural_net_deep_script.py
   :lines: 9-12

We create three data sets, corresponding to the official training,
validation, and test data splits of MNIST. For the training data set,
we set a batch size of 100 training examples, while the validation and
test data sets are used as complete batches.

Instantiating a model
---------------------

To train a model, you simply need to create an object representing a
model that inherits from the abstract base class
:class:`hebel.models.Model`. 

.. literalinclude:: ../examples/mnist_neural_net_deep_script.py
   :lines: 17-21

Currently, Hebel implements the following models:

* :class:`hebel.models.NeuralNet`: A neural net with any number of
  hidden layers for classification, using the cross-entropy loss
  function and softmax units in the output layer.

* :class:`hebel.models.LogisticRegression`: Multi-class logistic
  regression. Like :class:`hebel.models.NeuralNet` but does not have
  any hidden layers.

* :class:`hebel.models.MultitaskNeuralNet`: A neural net trained on
  multiple tasks simultaneously. A multi-task neural net can have any
  number of hidden layers with weights that are shared between the
  tasks and any number of output layers with separate weights for each
  task.

The :class:`hebel.models.NeuralNet` model we are using here takes as
input the dimensionality of the data, the number of classes, the sizes
of the hidden layers, the activation function to use, and whether to
use dropout for regularization. There are also a few more options such
as for L1 or L2 weight regularization, that we don't use here.

Training the model
------------------

To train the model, you first need to create an instance of
:class:`hebel.optimizers.SGD`:

.. literalinclude:: ../examples/mnist_neural_net_deep_script.py
   :lines: 23-32

First we are creating a :class:`hebel.monitors.ProgressMonitor`
object, that will save regular snapshots of the model during training
and save the logs and results to disk.

Next, we are creating the :class:`hebel.optimizers.SGD` object. We
instantiate the optimizer with the model, the parameter update rule,
training data, validation data, and the schedulers for the learning
rate and the momentum parameters.

Finally, we can start the training by invoking the
:meth:`hebel.optimizers.SGD.run` method. Here we train the model for
100 epochs. However, by default :class:`hebel.optimizers.SGD` uses
early stopping which means that it remembers the parameters that give
the best result on the validation set and will reset the model
parameters to them after the end of training.

Evaluating on test data
-----------------------

After training is complete we can do anything we want with the trained
model, such as using it in some prediction pipeline, pickle it to
disk, etc. Here we are evaluating the performance of the model on the
MNIST test data split:

.. literalinclude:: ../examples/mnist_neural_net_deep_script.py
   :lines: 37-40


