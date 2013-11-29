***************
Getting Started
***************

There are two basic methods how you can run Hebel:

#. You can write a YAML configuration file that describes your model
   architecture, data set, and hyperparameters and run it using the
   ``train_model.py`` script.
#. In your own Python script or program, you can create instances of
   models and optimizers programmatically.

The first makes estimating a model the easiest, as you don't have to
write any actual code. You simply specify all your parameters and data
set in an easy to read YAML configuration file and pass it to the
``train_model.py`` script. The script will create a directory for your
results where it will save intermediary models (in pickle-format), the
logs and final results.

The second method gives you more control over how exactly the model is
estimated and lets you interact with Hebel from other Python programs.


Running models from YAML configuration files
============================================

If you check the example YAML files in ``examples/`` you will see that the configuration file defines three top-level sections:

#. ``run_conf``: These options are passed to the ``run()`` method of
   the :class:`hebel.optimizers.SGD`-class.
#. ``optimizer``: Here you instantiate a :class:`hebel.optimizers.SGD`
   object, including the model you want to train and the data to use
   for training and validation.
#. ``test_dataset``: This section is optional, but here you can define
   test data to evaluate the model on after training.
