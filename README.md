dnn
===
This is my deep boltzmann machine (DBM) training for my Junior independent work.  This was mostly an educational exercise to learn more about how DBMs are implemented in practice and the decisions that go into training them.  If you are looking for a general deep learning framework, take a look at the pylearn2 project.


This is my second implementation because I realized the generally modular architecture makes a lot of sense.  A nice result is that it makes it easier to debug individual parts and reduces code duplication across different models.


Overall architecture is to have different kinds of **layers** (binary, gaussian, etc.) and **connections**
between them that handle propagating activations.  For now these are fully
connected and [http://arkitus.com/ShapeBM/](ShapeBM) style connection, but there is no reason it wouldn't work for convolutions as well.  I tried to make it easy to add different kinds of model statistics and update methods, but the specific methods are still fairly purpose built.  The only training method right now is minibatch stochastic gradient descent.

Each model is made up of a list of layers, list of connections, and any statistics that should be run (generally at least a data dependent and model dependent statistic).

Dependencies
=============
+ PIL (or Pillow) - Used for image processing
+ Matplotlib - used for visualization
+ Numpy - used for matrix calculations



Usage
======

Basic usage and examples are illustrated in the ipython notebook
[http://nbviewer.ipython.org/github/dmrd/dnn/blob/master/demo.ipynb](demo.ipynb).
