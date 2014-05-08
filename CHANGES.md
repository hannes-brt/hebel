Hebel Changelog
===============

Version 0.02
------------

05-08-2014

* Windows compatibility (Thanks to @Wainberg)
* CUDA 4.x is no longer supported, please upgrade to CUDA 5 or CUDA 6
* All initialization is now handled through `hebel.init()`. No need to
  initialize PyCUDA separately anymore.
* `LogisticLayer` has been renamed to `SoftmaxLayer`. `LogisticLayer`
  now does binary classification while `SoftmaxLayer` is for
  multiclass classification.
* Framework for cross-validation.
* When `ProgressMonitor` has `save_interval=None`, then only the
  currently best model is serialized. If it is a positive integer,
  then regular snapshots of the model are stored with that frequency.

Version 0.01
------------

01-01-2014

* Removed dependency on scikits.cuda (this should make Hebel
  compatible with Windows, but I couldn't test that yet)

* Serious speed-ups by avoiding freeing and reallocating memory for
  temporary objects. Previously, many temporary gpuarrays were
  reallocated in every single minibatch and then discarded, which was
  very inefficient. By using persistent objects for temporary objects
  across minibatches and some other improvements such as doing more
  computations in-place, a roughly 2x speed-up could be realised.
