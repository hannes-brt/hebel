***************
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

The next two sections define the data for the model. All data must be given as instances of ``DataProvider`` objects:

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
