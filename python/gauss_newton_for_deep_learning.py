#########################################################################
# Equations are in here:                                                #
# Shen 2017, Towards a Mathematical Understanding of the Difficulty in  #
# Learning with Feedforward Neural Networks                             #
# https://arxiv.org/abs/1611.05827                                      #
#########################################################################

import logging

logging.root.handlers = []  
logging.basicConfig(level = logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n')

logger = logging.getLogger(__name__)

import numpy as np
import scipy as sp
import scipy.io
import scipy.linalg

from matplotlib import pyplot as plt

import os
import sys
import copy
import time

from pprint import pprint as pp
from collections import OrderedDict
from progressbar import ProgressBar as progbar
from itertools import  *

from common import *

from fourregionwrapper import FourRegionWrapper
from xorwrapper import XORWrapper

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global variables
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_error(run, errors):
  fig = plt.figure()
  fig.canvas.set_window_title(run)
  plt.plot(errors, "-b")
  plt.xlabel("iterations")
  plt.ylabel("error")
  plt.title("Test set error over time")
  #plt.yscale("log")
  plt.grid()
  return
def plot_weight_distance(run, weights):
  ''' 
  Plots the distance from the last weight in the list to show the convergence speed. The
  distance is meassured by the l2 norm
  '''
  
  weights_no_none = [W[1:] for W in weights]
  weights_flat = map(unfold_data_numpy, weights_no_none)

  W_star = weights_flat[-1]
  W_distance = [ np.linalg.norm(W - W_star) for W in weights_flat]

  fig = plt.figure()
  fig.canvas.set_window_title(run)
  plt.plot(W_distance, "-b")
  plt.xlabel("iterations")
  plt.ylabel("distance")
  plt.title("distance from ground truth")
  plt.yscale("log")
  plt.grid()

  return

def calculate_F(x,params, non_linearities, prime=False):
  '''
  Runs the foward pass through the network for the given input x.
  The output of all layers is returned as list.
  
  The argument prime controls whether or not the derivative mappings are invovled.

  This method assumes the usage of extended weight matrices W and dummy units
  '''

  # Number of layers (including input)
  L = len(params)

  # Copy of input to avoid call by reference issues
  output = [x.copy()] + [None] * (L - 1)

  # Stores the output of the layer, where the layer at position l gets the derivative
  # of the activation function (the input to this derivative layer mapping is the normal
  # forward pass)
  output_prime = [x.copy()] + [None] * (L - 1)

  for i,W in enumerate(params):
    
    # Skip input layer
    if i == 0:
      continue

    # Append dummy unit to layer input
    phi = np.append(output[i - 1], 1.0)
        
    z = np.dot(W.T, phi)

    output[i] = non_linearities[i][0](z)
    output_prime[i] = non_linearities[i][1](z)

  #end for

  return output if not prime else output_prime

def l2_norm(x,y):
  ''' Std euclidean norm '''
  assert x.shape == y.shape, "Shape missmatch: {} and {}".format(x.shape, y.shape)
  return np.linalg.norm(x - y)
def G1_l2_norm(x,y):
  ''' Gradient of l2 norm wrt. the first argument '''
  assert x.shape == y.shape, "Shape missmatch: {} and {}".format(x.shape, y.shape)
  return x - y
def H1_l2_norm(x,y):
  ''' Hessian of l2 norm wrt. the first argument '''
  assert x.shape == y.shape, "Shape missmatch: {} and {}".format(x.shape, y.shape)
  #return np.eye(len(x), len(x))
  return np.diag(np.ones_like(x))

def calculate_train_error(params, non_linearities,data_wrapper):
  ''' Calculates the training error '''
  loss = 0.0

  for i in xrange(data_wrapper.X_train.shape[0]):
    x = data_wrapper.X_train[i]
    y = data_wrapper.Y_train[i]

    output = calculate_F(x, params, non_linearities, False)[-1]

    loss += l2_norm(output,y)
  #end

  return loss
def calculate_test_error(params, non_linearities,data_wrapper):
  ''' Calculates the test error '''
  loss = 0.0

  for i in xrange(data_wrapper.X_test.shape[0]):
    x = data_wrapper.X_test[i]
    y = data_wrapper.Y_test[i]

    output = calculate_F(x, params, non_linearities, False)[-1]

    loss += l2_norm(output,y)
  #end

  return loss

def fin_diff(params,x,y, i=0,j=0):
  '''
  Verification via finite differences, assumes std. inner product 
  in vanilla euclidean space.

  This code is just a quick verification and is most likely 
  NOT numerically robust.
  '''
  eps = 1e-6

  direction = np.zeros_like(params[-1])
  direction[i,j] = 1.0
  
  params_inc = copy.deepcopy(params)
  params_inc[-1] += eps * direction
  f_inc = calculate_F(x, params_inc, non_linearities)
  
  params_dec = copy.deepcopy(params)
  params_dec[-1] -= eps * direction
  f_dec = calculate_F(x, params_dec, non_linearities)

  deriv = (l2_norm(f_inc[-1], y) - l2_norm(f_dec[-1], y)) / (2.0 * eps)
  #print "{:3e}".format(deriv)
  return deriv

def get_gradient_wrt_W_for_sample_old(params, phi, psi, y):
  '''
  Returns the Gradient for the current sample, this method is based on 
  ealier calculations and serves only as a reference
  '''

  # The number of layers including the input layer   
  L = len(params)

  # The gradient of the error function, i.e. E(x,y) = 0.5 || x - y ||^2, wrt. the first argument x (F_L or phi_L)   
  G_E_1 = G1_l2_norm(phi[L - 1], y) 

  # The gradient for the weights (see eq 14, 15 & 16) in correct order, first element is None for zero-th input layer 
  G_J_W = [None] + [ np.dot(np.outer(np.append(phi[l - 1],1.0), G_E_1) , psi[l].T) for l in xrange(1,L)]

  # Now stack all flattened elements to the full gradient. In this step also the first element is removed.
  # Be aware that flatten has to be applied to the transposed gradient in order to match the Hessian (or use fortran style)
  G_J_W_flat = np.concatenate(map(lambda foo: foo.T.flatten(), G_J_W[1:]))

  return G_J_W_flat
def get_gradient_wrt_W_for_sample(params, phi, psi, y):
  ''' 
  Returns the Gradient for the current sample
 
  Compare with eq. 18:  
  the Kronecker product is performed in the code with an identity matrix first before mutliply psi from the left
  for debugging reasons
  '''
  
  # The number of layers including the input layer   
  L = len(params)

  # These are also the shapes of the directions H_k
  param_shapes = shapes(params[1:])

  # The gradient of the error function (i.e. E(x,y) = 0.5 || x - y ||^2) wrt. the first argument x 
  # Here, x is F_L or phi_L, i.e. the forward pass through the network
  G_E_1 = G1_l2_norm(phi[L - 1], y) 

  # Block diagonal matrices that are used to vectorize the directions
  phi_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi)]
  
  # The gradients for every weight matrix W_l in correct order, first element is None for zero-th input layer which
  # has no weights (only for convenience)
  G_J_W = [None] + [np.linalg.multi_dot([G_E_1, psi[l].T, phi_block[l - 1]])  for l in xrange(1,L) ]

  # Now stack all flat gradients together to the full one. In this step also the first element is removed.
  # Due to the definition, this gradient is a lying row vector ( for the inner product with the vectorized direction )
  # -> This is a 1d array so transposed definition does not matter
  G_J_W_flat = np.concatenate(G_J_W[1:])

  return G_J_W_flat
def get_hessian_wrt_W_for_sample(params, phi, psi, y):
  ''' 
  Returns the Hessian for the current sample, this method is based on 
  ealier calculations and serves only as a reference

  Compare with eq. 27
  '''

  # The number of layers including the input layer   
  L = len(params)

  # These are also the shapes of the directions H_k
  param_shapes = shapes(params[1:])
  
  # The hessian of the error function wrt. twice the first argument x (i.e. F_L or phi_L)  
  H_E_1 = H1_l2_norm(phi[L - 1],y)

  # The blocks for the inner matrix when expanding the sum in eq. 27
  psi_tilde = [[ np.linalg.multi_dot([psi_i, H_E_1 , psi_j.T]) for psi_j in psi[1:] ] for psi_i in psi[1:]]

  # The single large matrix psi * H_E * psi.T for all indices
  psi_tilde_full = np.vstack(map(np.hstack, psi_tilde))

  # The list of block diagonal phi matrices for all H_k, phi is also extended with a dummy unit
  # The transpose has no effect (1d arrays) and is there to match the equations
  phi_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi)]

  # The single large block diagonal matrix which can be multiplied to psi_tilde_full from both sides
  phi_block_full = sp.linalg.block_diag(*phi_block)

  # Now build the final full sized Hessian matrix
  H_J_W = np.linalg.multi_dot([phi_block_full.T, psi_tilde_full, phi_block_full])

  return H_J_W

def get_gradient_and_hessian_wrt_W_for_sample(params, non_linearities, x,y, no_hessian, **kwargs):
  ''' Returns the Gradient and Hessian wrt. W for the current sample '''
  
  # The number of layers including the input layer   
  L = len(params)
  
  # Collect the layer mappings
  # see eq. 3 & 4
  F = calculate_F(x, params, non_linearities, False)

  # Collect the layer derivative mappings
  # see eq. 8 & 9
  F_prime = calculate_F(x, params, non_linearities, True)

  # Renaming to match notation in paper
  phi = F
  
  # The recursive definition of psi_l (see eq 17). The last matrix Sigma_L has the index L-1 in the code. Be carful here!
  psi = [None] * (L - 1) + [np.diag(F_prime[L - 1])]

  # Iterate backwards as in paper
  for l in xrange(L - 2,-1,-1):

    # In eq. 13 the direction h_l is applied to the layer input phi_l, thus the vector ends with an zero instead of one.
    # -> this indicates a direction in homogenous coordinates
    # -> the zero simply removes the last column in the transposed weight matrix (i.e. the bias vector)
    #    W.T[:,0:-1] == W[0:-1,:]
    rhs = np.dot(params[l + 1][0:-1,:], psi[l + 1])

    # Then multiply with the next sigma to get psi_l
    psi[l] = np.dot(np.diag(F_prime[l]) , rhs)

  #end for
  
  G_J_W = get_gradient_wrt_W_for_sample(params,phi, psi, y)

  if no_hessian:
    H_J_W = 0.0 # <- gets broadcasted correctly

  else:    
    H_J_W = get_hessian_wrt_W_for_sample(params, phi, psi, y)
  
  return G_J_W, H_J_W

def build_gradient_and_hessian_wrt_W(params, non_linearities,data_wrapper, **kwargs):
  ''' Calculates the complete gradient and hessian, i.e. performs the sum over all samples (eq. 36) '''
  
  N_net = count_parameter(params)

  H_W = np.zeros((N_net, N_net))
  G_W = np.zeros((N_net,))

  T = len(data_wrapper)

  for i in xrange(T):
    G,H = get_gradient_and_hessian_wrt_W_for_sample(params, non_linearities, data_wrapper.X_train[i],data_wrapper.Y_train[i], **kwargs)
    G_W += G
    H_W += H
  #end for
  
  return G_W, H_W
def descent_step(params, non_linearities, data_wrapper,alpha, delta, armijo_c, no_hessian, **kwargs):
  ''' A single training step '''
  
  W_shapes = shapes(params[1:])
  
  G_W, H_W = build_gradient_and_hessian_wrt_W(params, non_linearities,data_wrapper,  no_hessian = no_hessian, **kwargs)

  if no_hessian is True:
    params_updates_flat = G_W

  else:
    H_W_reg = H_W + delta * np.eye(*H_W.shape)
    params_updates_flat = np.linalg.solve(H_W_reg,G_W)
  #end if

  # Convert flat array to list of matrices
  params_updates = fold_data_numpy(params_updates_flat, W_shapes, to_float32 = False)
  
  # Simple descent step
  params_new = [None] + [W - alpha * W_dir for W, W_dir in izip(params[1:], params_updates)]
  error_new = calculate_test_error(params_new, non_linearities, data_wrapper)

  # Run a line search with armijo condition
  # Untested, leftover from earlier tests

  #error_cur = calculate_train_error(params, non_linearities)
  #grad_times_dir = np.dot(G_W, params_updates_flat)

  #armijo_condition = False
  #alpha = 1.0
  #counter = 0
  
  #while not armijo_condition and counter < 100:
  #  params_new = [None] + [W - alpha * W_dir for W, W_dir in zip(params[1:], params_updates)]
  #  error_new = calculate_train_error(params_new, non_linearities)
  #  armijo_condition = error_new <= error_cur - armijo_c * alpha * grad_times_dir

  #  alpha *= 0.5 
  #  counter += 1

  #  if alpha < 1e-9:
  #    break
  ##end for line search

  #if armijo_condition:
  #  logger.info("Linesearch succeeded after {} steps, alpha = {}".format(counter,alpha * 2.0))
  #else:
  #  logger.warning("Linesearch failed, alpha = {}".format(alpha * 2.0))
  ##end

  return params_new, error_new      
def train(params, non_linearities, data_wrapper, run, iterations, plotting_rate,show_progress, **kwargs):
  ''' The main training loop, returns the lists with parameters and errors over time '''

  t0 = time.time()

  errors_all = []
  params_all = []

  it = xrange(iterations)

  if show_progress:
    it = progbar()(it)

  for i in it:

    params, error = descent_step(params, non_linearities, data_wrapper, **kwargs)
    
    errors_all.append(error)
    params_all.append(params)

    if plotting_rate > 0 and (i + 1) % plotting_rate == 0:
      plot_error(run, errors_all)
      plot_weight_distance(run, params_all)
      plt.show()
    #end if

    # Relative change requires more than one error
    if len(errors_all) < 2:
      continue

    relative_change = np.abs(errors_all[-1] - errors_all[-2]) / np.abs(errors_all[-2])

    threshold = 1e-25
    if relative_change < threshold:
      logger.info("The relative change of the error is smaller than {} after {} iterations, terminating.".format(threshold, i))
      break
    #end

    threshold = 1e100
    if relative_change > threshold:
      logger.info("The relative change of the error is larger than {} after {} iterations, terminating.".format(threshold, i))
      break
    #end

  #end for

  t1 = time.time()

  return params_all, errors_all, t1 - t0

def parse_arguments(*args,**kwargs):
  ''' Creates the parser and returns the parsed result '''
  import argparse
  parser = argparse.ArgumentParser(description = "Gauss Newton for FNN Training")

  parser = parser_add_common_arguments(parser)

  task_names = [ k for k in globals().keys() if k.startswith("task_")]
  parser.add_argument('-t', '--task', type = str, choices = task_names, default = "task_dummy",
                      help = "The task to run, i.e. one of the functions starting with 'task_'. Default is %(default)s.")

  parser.add_argument('-c', '--armijo_c', default = 0.99, type = float, help = "The constant c that is used in the armijo condition, typically close to 1.0. Default is %(default)s")

  args,unkowns = parser.parse_known_args(*args)

  if len(unkowns) != 0:
    logger.warn("Unrecognized arguments are present: {}".format(unkowns))

  parser_finalize(args,**kwargs)

  return args

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# These functions represent the tasks that can be started
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def task_dummy(args, **kwargs):
  ''' A dummy task, which is the dault value for the arg parser '''
  return
def task_xor(args):
  ''' This task trains a classifier for the XOR classification problem. Quadratic convergence can be observed. '''

  # Overwrite settings with required parameters for this task
  args.alpha = 1.0
  
  data_wrapper = XORWrapper()

  params_init, non_linearities = create_network(**vars(args))
  
  params, errors, timings = the_time_saver(train, os.path.join(build_dir,args.run), args.recreate,
                                           params_init, non_linearities, data_wrapper, **vars(args))

  # The xor problem is a bit different than the four region classifier, since it is actually a regression problem
  xor_classifier = lambda x: calculate_F(x, params[-1], non_linearities, False)[-1]
  y_classes = np.array(map(xor_classifier, data_wrapper.X_test))
  y_classes_int = np.round(y_classes).astype(np.int)
  y_classes_sorted = data_wrapper.sort_data_set(data_wrapper.X_test, y_classes_int)

  data_wrapper.plot_data_set(args.run, y_classes_sorted, title = "Gauss newton XOR FNN classifier")
  data_wrapper.plot_training_data_set()
  data_wrapper.plot_solution(args.run)

  plot_weight_distance(args.run, params)
  plot_error(args.run,errors)

  plt.show()

  return params, errors, timings
def task_fourregion(args):
  ''' This task trains a classifier on the four region classification problem '''

  data_wrapper = FourRegionWrapper(use_file = False, T = 1000)

  params_init, non_linearities = create_network(**vars(args))
  
  params, errors, timings = the_time_saver(train, os.path.join(build_dir,args.run), args.recreate,
                                           params_init, non_linearities, data_wrapper, **vars(args))

  classifier = lambda x: calculate_F(x, params[-1], non_linearities, False)[-1]
  
  data_wrapper.plot_training_data_set()
  data_wrapper.plot_solution(args.run)
  data_wrapper.plot_classifier("My Gauss Newton 4 Region Classifier", classifier)

  plot_weight_distance(args.run, params)
  plot_error(args.run,errors)

  plt.show()

  return params, errors, timings
def task_fourregion_quad_conv(args):
  ''' This task demonstrates the quadratic convergence in a simplified four region classification problem '''

  # Overwrite settings with required parameters for this task
  args.alpha = 1.0
  args.iterations = 100

  # Create a tiny dataset where exact learning is possible
  data_wrapper = FourRegionWrapper(use_file = False, T = 10)

  params_init, non_linearities = create_network(**vars(args))
  
  params, errors, timings = the_time_saver(train, os.path.join(build_dir,args.run), args.recreate,
                                           params_init, non_linearities, data_wrapper, **vars(args))

  plot_weight_distance(args.run, params)
  plot_error(args.run,errors)

  plt.show()

  return params, errors, timings

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code to run the stuff
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":

  # Uncomment one of the following lines and let the code run.
  # This is the same as using the arg parser from the command line.
  # For a list and explanation of possible arguments please refer to the help 
  # of the arg parser

  # Trains a classifier in the large four region problem, takes around 30 minutes
  # Regularisation must be stronger than default (1e-8)
  args = parse_arguments("run_1 -t task_fourregion -n 2 10 10 4 -a 0.01 -rw -i 1000 -d 1e-5 -sp".split(" "))

  # Trains a classifier for the XOR prolem, very fast. Demonstrates the quadratic convergence.
  #args = parse_arguments("run_2 -t task_xor -n 2 4 2 1 -rw -a 1 -i 1000 -sp".split(" "))

  # Trains a classifier for the 4 region prolem. Demonstrates the quadratic convergence.
  #args = parse_arguments("run_3 -t task_fourregion_quad_conv -n 2 10 10 4 -a 1 -rw -i 100 -sp".split(" "))

  # If argument have been specified in the command line the hard coded ones are overwritten
  if len(sys.argv) > 1:
    args = parse_arguments()
      
  logging.info("Launching task: {}".format(args.task))

  tasks = { k:v for k,v in locals().iteritems() if k.startswith("task_")}
  results = tasks[args.task](args)
    
  params, errors, timings = results
  
  print "Elapsed time: ", timings
   
