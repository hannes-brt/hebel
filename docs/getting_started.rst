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
`train_model.py`` script. The script will create a directory for your
results where it will save intermediary models (in pickle-format), the
logs and final results.

The second method gives you more control over how exactly the model is
estimated and lets you interact with Hebel from other Python programs.


Running models from YAML configuration files
============================================

.. autoclass:: hebel.models.neural_net.NeuralNet
   :members:
