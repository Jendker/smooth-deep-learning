#########################################################
# This script contains code snippets, which are used in #
# different scripts                                     #
#########################################################

import logging
logger = logging.getLogger(__name__)

import numpy as np

import os
import operator
import cPickle

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global variables
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# The current working directory, can be relative to the scipt with the entry point, or can be an
# absolut path
working_dir = "."

# The subfolder where results and temporary files are stored
build_dir = os.path.join(working_dir, "build")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Sigmoid activation function
sigma = lambda t: 1.0 / (1.0 + np.exp(-t))

# The derivative of the sigmoid function, 1.0 - sigma works elementwise for the batch
d_sigma = lambda t: sigma(t) * (1.0 - sigma(t))

# The second derivative of the sigmoid function, currently not used
# dd_sigma = lambda t: d_sigma(t) * (1.0-sigma(t)) - sigma(t)* d_sigma(t) -> from productrule, simplified:
dd_sigma = lambda t: d_sigma(t) * (1.0 - 2.0 * sigma(t))

# Bend identity activation function
bent_id = lambda t: 0.5 * (np.sqrt(t ** 2 + 1.0) - 1.0) + t

# Derivative of the bend identity activation function
d_bent_id = lambda t: t / (2.0 * np.sqrt(t ** 2 + 1.0)) + 1.0

# Linear activation function
linear = lambda t: t

# The constant derivative of a linear activation function
d_linear = lambda t: np.ones_like(t)

# Overwrites the builtin sum to avoid a problem with adding stuff to int 
# (first element in builtin sum is always 0)
sum = lambda seq: reduce(operator.add, seq)

# Similar to sum, but with multiplication
mul = lambda seq: reduce(operator.mul, seq)

# Can be mapped to a list of numpy arrays
shape = lambda foo: foo.shape if foo is not None else (-1,)

# Maps the shape extractor to the given list
shapes = lambda foo: map(shape, foo)

def the_time_saver(function, filename, make_new, *args, **kwargs):
  '''
  First tries to load the result of the function from the file.
  If this failes, the function is called with the args and kwargs and its result is pickled.
  '''

  try:
  
    # Manually raise this error to trigger recreation
    if make_new:
      raise IOError

    with open(filename,'rb') as the_file:
      result = cPickle.load(the_file)
    #end with

    logger.info("Returning results for {} from file.".format(os.path.split(filename)[-1]))

  except IOError as e:

    # Create results from scratch
    result = function(*args, **kwargs)

    with open(filename,'wb') as the_file:
      cPickle.dump(result, the_file, protocol = cPickle.HIGHEST_PROTOCOL)
    #end
    
    logger.info("Returning results for {} from function.".format(os.path.split(filename)[-1]))

  # end

  return result
def dummy(*args, **kwargs): 
  ''' 
  Dummy function to be used in my time saver. If the file cannot be loaded a RuntimeError is raised 
  so that the pickled results are not overwritten.
  '''
  raise RuntimeError("File could not be loaded")
  
def fold_data_numpy(data, data_shapes, to_float32=True):
  '''
  Folds the data for usage in the network (e.g. updating the list of trainable weights)

  The data is assumed to be a 1d numpy array with shape ( num_params, ) and is converted 
  to a list of the form [ W1, W2, ....], where W_l is the weight matrix of layer l.
     
  The argument 'data_shapes' provides the required information for folding the array

  If enabled, this function applies a type conversion to float32 which is required by 
  the CUDA gpu arrays. 

  IMPORTANT: unfolding has to be done in the "F" style (Fotran), otherwise folding and unfolding 
             won't result in the original data.
  '''
  N = len(data_shapes)
           
  data_folded = [None] * N
    
  start = stop = increment = 0

  for i in xrange(N):
    increment = np.prod(data_shapes[i]) # Number of elements in matrix or vector
    stop = start + increment

    data_folded[i] = data[start:stop].reshape(data_shapes[i], order = "F")
    
    if to_float32:
      data_folded[i] = data_folded[i].astype(np.float32)

    start += increment
  #end for

  return data_folded
def unfold_data_numpy(data, style="C", to_column_vector=False, num_params=None):
  '''
  Unfolds the data which comes from the network (e.g. the list of trainable weights)

  The data is a list of the form [ W1, W2, ...], where W_l is the weight matrix of layer l.
  
  The shape of weight matrices is (num_incoming + 1, num_units)
   
  The returned vector is a flat array with the shape ( num_params, ).

  If the number of parameters is provided via the argument 'num_params', a simple check
  is performed to make shure the flattened data has the correct length.

  By setting 'to_column_vector' to True, the resulting flat array is reshaped to a true 
  column vector.

  The argument 'style' determines, wether rows or columns of the data are concatenated:

  >>> x.flatten("F") == x.T.flatten("C")
  >>> True
  '''

  if len(data) <= 0:
    raise ValueError("Recieved empty list")
  
  if style not in ["C","F"]:
    raise ValueError("The style {} is not supported".format(style))
  
  data_unfolded = np.concatenate(map(lambda foo: foo.flatten(style), data))

  # Reshape to turn it in column vector (if desired)
  if to_column_vector:
    data_unfolded = data_unfolded.reshape((-1,1))

  # Final check on shape if the number of parameters is provided
  if num_params is not None:
    if to_column_vector:
      assert data_unfolded.shape == (num_params, 1), "The produced unfolded data vector has the wrong shape"
    else:
      assert data_unfolded.shape == (num_params,), "The produced unfolded data vector has the wrong shape"
    #end
  #end

  return data_unfolded

def parser_add_common_arguments(parser):
  ''' 
  This functions adds to the given ArgumentParser instance common arguments,
  which are used in several scripts.
  '''

  import argparse
  assert isinstance(parser, argparse.ArgumentParser)

  parser.add_argument('run', type = str,
                      help = "A trivial name of the current run, it is used as part of filenames for storing the results of this run.")

  parser.add_argument('-a', '--alpha', default = 1e-1, type = float,
                      help = "The initial stepsize for the line search, or the learning rate in general. Default is %(default)s.")
  parser.add_argument('-n', '--num_units', nargs = '+', dest = "n", default = [2,10,10,1], type = int,
                      help = "Defines the network layout by specifing the number of units in each layer. The length of the list is thus the depth of the network. Default is %(default)s.")
  parser.add_argument('-d', '--delta', default = 1e-8, type = float,
                      help = "The strength of the regularization of the Hessian. Default is %(default)s.")
  parser.add_argument('-p', '--plotting_rate', default = -1, type = int, 
                      help = "Defines the number of training steps after which the errors are plotted, -1 to disable. Default is %(default)s.")
  parser.add_argument('-i', '--iterations', default = 15000, type = int,
                      help = "The number of epochs for training. Default is %(default)s.")
  parser.add_argument('-T', '--trajectory_length', default = 1000, type = int, dest = 'T', 
                      help = "The number of samples in the training set or the length of the trajectories, i.e. the number of (s,a,r,s',a') tuples collected in a roll-out. Default is %(default)s.")
  parser.add_argument('-fw', '--filename_weight', default = "init_weights_{run}_{n}.pickle", type = str,
                      help = "The filename of the initial weights for pickling, will be formatted with the named arguments 'run' and 'n'."
                             " Thus 'foo_{run}_{n}.pickle' or 'foo_{run}.pickle' or 'foo_{n}.pickle' are possible. Default is %(default)s.")

  parser.add_argument('-g', '--use_glorot', action="store_true", 
                      help = "If specified, weights are initialized according to Glorot et al.")
  parser.add_argument('-r', '--recreate', action="store_true", 
                      help = "If specified, the training and evaluation part of a task is redone despite existing pickle files. See the 'make_new' argument of the time saver.")
  parser.add_argument('-rw', '--random_weights', action="store_true",
                      help = "If specified, the weights are initialized randomly. Else the weights have to be loaded from file.")
  parser.add_argument('-rd', '--random_data', action="store_true",
                      help = "If specified, the data is collected from randomized roll-outs. Else a file with the data has to be provided.")
  parser.add_argument('-nh', '--no_hessian', action="store_true",
                      help = "If specified, the Hessian is not calculated -> Classic Backpropagation.")
  parser.add_argument('-ns', '--no_saving', action="store_true",
                      help = "If specified, the saving of the arg parser object is disabled. Useful to test different arguments for the same run without overwriting existing stuff.")  
  parser.add_argument('-sp', '--show_progress', action="store_true", 
                      help = "If specified, progress bars are shown.")    

  return parser
def parser_finalize(args,**kwargs):
  ''' 
  Store the arguments under the run name in the build folder for reproduction, if enabled.
  The variable 'args' without star is correct.
  '''
  
  if kwargs.get("save_args", True) and not args.no_saving:
    with open(os.path.join(build_dir, "arguments_{}.pickle".format(args.run)), "wb") as the_file:
      cPickle.dump(args, the_file, cPickle.HIGHEST_PROTOCOL)
  
  return

def init_params(n, use_glorot=False, **kwargs):
  ''' 
  Creates a network, where the bias is presented with larger weight matices and dummy units.

  Intializes the parameters of a FNN from the list n, which contains the number of units in each layer. 
  n[0] is the input dimension, n[-1] the output dim.
  '''

  # No parameters for the input layer
  params = [None]
  
  for l in xrange(len(n) - 1):

    if use_glorot:
      # Glorot Initializer
      fan_in = 1.0 / np.sqrt(n[l])
      W = np.random.uniform(-fan_in,fan_in, size = (n[l] + 1, n[l + 1]))
      W[-1,:] = 0.0
    else:
      W = np.random.normal(size = (n[l] + 1, n[l + 1]))
    #end if

    params.append(W)
  #end for

  # Code to load matlab matrices for comparing code
  #import scipy.io
  #foo = scipy.io.loadmat("weights.mat")["WeightCell"]
  #
  #W1 = foo[0,0]
  #W2 = foo[0,1]
  #W3 = foo[0,2]
  #
  #params = [None] + [W1,W2,W3]

  return params
def count_parameter(params):
  ''' 
  Returns the number of parameters i.e. the sum of multiplied shapes of weight matrices W.
  Assumes that the first entry is [None] for the input layer
  '''
  return sum(map(mul, shapes(params[1:])))
def create_network(run,n,random_weights, filename_weight, recreate, **kwargs):
  '''
  Calls the method for the actual initialization of parameters, see its description for details.

  The list of non-linearities is created here (currently hardcoded)

  If recreate_weights is true, existing pickled weights are ignored and they are again initialized randomly.
  This argument works only if random iniital weights are used.
  
  Without randomized initial weights, the weights have to be unpickled.
  '''
   
  filename = filename_weight.format(run = run, n = str.join("_",map(str,n)))

  if random_weights:
    params = the_time_saver(init_params, os.path.join(build_dir,filename), recreate,
                            n,**kwargs)
  else:
    params = the_time_saver(dummy, os.path.join(build_dir,filename), False)
  #end

  # The number of layers
  L = len(n)

  # No non-linearity for the input layer
  # Hidden layers get the bent identity (there are L-2 of them)
  # Output is linear
  non_linearities = [None] + [(bent_id, d_bent_id) for i in xrange(L - 2) ] + [(linear, d_linear)]

  return params, non_linearities