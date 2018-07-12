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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rc("font", size = 7)

from itertools import  *
import os
import sys
import cPickle
import time
from pprint import pprint as pp
from fnmatch import fnmatch as match
from progressbar import ProgressBar as progbar
from collections import OrderedDict

from environmentwrapper import *

from common import *

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global variables
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Variables with the filenames -> autocompletion in code
file_mountaincar_data_fixed_policy = "mountaincar_data_fixed_policy.pickle"
file_mountaincar_data_off_pol_pe = "mountaincar_data_off_pol_pe.pickle"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Short version for showing plots in interactive shell
shw = plt.show

def calculate_weight_distance(weights):

  weights_no_none = [W[1:] for W in weights]
  weights_flat = map(unfold_data_numpy, weights_no_none)

  W_star = weights_flat[-1]
  W_distance = [ np.linalg.norm(W - W_star) for W in weights_flat]

  return W_distance

def calculate_Q(s,a,params, non_linearities, prime=False):
  '''
  Runs the foward pass through the Q network for the given input (s,a). 
  The output of all layers is returned as list.
  
  The argument prime controls whether or not the derivative mappings are invovled.

  This method assumes the usage of extended weight matrices W and dummy units
  '''

  # Combine the continuous state vector and the single discrete action to
  # single vector
  x = np.append(s,a)
  
  # Use it with the normal value function network
  return calculate_V(x, params, non_linearities, prime)
def calculate_V(s,params, non_linearities, prime=False):
  '''
  Runs the foward pass through the value function network for the given state s. 
  The output of all layers is returned as list.
  
  The argument prime controls whether or not the derivative mappings are invovled.

  This method assumes the usage of extended weight matrices W and dummy units
  '''

  # Number of layers (including input)
  L = len(params)

  # Renaming to use same code as before
  x = s

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

def evaluate_V(params, non_linearities, data_wrapper):
  ''' 
  Evaluate the value-function network at the discretized states
  that are used in the Monte Carlo version 
  '''
    
  V_net = np.empty((data_wrapper.binsX, data_wrapper.binsY))

  #for idx, (s,a) in data_wrapper.iterate_state_action_space():
  for idx, s in data_wrapper.iterate_state_space():
    V_net[idx] = calculate_V(s,params, non_linearities)[-1]

  return V_net
def evaluate_distance_V_and_V_MC(V_MC, params, non_linearities, data_wrapper, steps=10, show_progress=False):
  '''
  Calculates the distance from the ground truth for all parameters
  Currently only every 10-th parameter vector is used.
  '''

  dist = []

  it = params[::steps]

  if show_progress:
    it = progbar()(it)

  for param in it:
    V_eval = evaluate_V(param, non_linearities, data_wrapper)
    d = np.linalg.norm(V_eval - V_MC)
    dist.append(d)
      
  return dist

def TD_error(Q, Q_prime, r , gamma):
  ''' 
  Q is the Q-function evaluated at state s and action a (i.e. forward pass)
  Q_prime is the same for s_prime and a_prime
  r is the reward associated to the (s,a,s_prime) tuple
  gamma is the discount factor
  
  Deterministic transitions 
  -> only one successor state s' has to be considered 
  -> only two forward passes through the network are required
  '''
  return Q - (r + gamma * Q_prime)
def TD_error_sq(Q, Q_prime, r , gamma):
  ''' 
  The squared TD error with prefactor
  '''
  return 0.5 * TD_error(Q,Q_prime,r,gamma) ** 2

def calculate_error(params, non_linearities, data_wrapper, Q_factors, Q_DQN, **kwargs):
  ''' Calculates the sum of squared TD-errors '''

  loss = 0.0
  T = len(data_wrapper)

  for i in xrange(T):
    
    s,a,r,s_prime,a_prime = data_wrapper[i]

    _a = np.int(a)
    _a_prime = np.int(a_prime)

    if Q_factors:      
      Q_sa = calculate_Q(s,a, params, non_linearities, False)[-1]
      Q_sa_prime = calculate_Q(s_prime, a_prime, params, non_linearities, False)[-1]
    
      loss += TD_error_sq(Q_sa, Q_sa_prime, r, data_wrapper.gamma)
    elif Q_DQN:
      Q_sa = calculate_V(s, params, non_linearities, False)[-1]
      Q_sa_prime = calculate_V(s_prime, params, non_linearities, False)[-1]
      
      loss += TD_error_sq(Q_sa[_a], Q_sa_prime[_a_prime], r, data_wrapper.gamma)
    else:
      V_s = calculate_V(s, params, non_linearities, False)[-1]
      V_s_prime = calculate_V(s_prime, params, non_linearities, False)[-1]
    
      loss += TD_error_sq(V_s[0], V_s_prime[0], r, data_wrapper.gamma)
    #end if

  #end for

  loss /= T
  
  return loss

def get_gradient_wrt_W_for_sample(params, phi, phi_prime, psi, psi_prime, r, gamma, constant_target, Q_DQN, a, a_prime):
  ''' 
  Returns the Gradient for the current (s,a,r,s',a')-tuple 
  
  Compare with eq. 18:
  - nabla_E (phi_L) is the Gradient of 1/2 delta^2 wrt. the first component only, i.e. delta
  - the Kronecker product is performed in the code with an identity matrix first before mutliply psi from the left
    for debugging reasons
  '''

  # The number of layers including the input layer   
  L = len(params)

  # These are also the shapes of the directions H_k
  param_shapes = shapes(params[1:])

  # For DQN layout Q is a vector whose components give the expected reward for each action in the given state
  # Therefore a "picking" operation has to be considered to generate the flat gradient, 
  # The entry is defined by the executed actions a (or a_prime)
  # For non-DQN layouts we have a true row vector -> select this one always with zero (same as calling np.squeeze(...))
  if Q_DQN:
    _a = a
    _a_prime = a_prime
  else:
    _a = 0
    _a_prime = 0
  #end
  
  # Gradient of the error function nabla_1 E(phi_L): 0.5 delta**2 results in delta
  TD = TD_error(phi[L - 1][_a], phi_prime[L - 1][_a_prime], r, gamma)

  # Block diagonal matrices that are used to vectorize the directions
  phi_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi)]

  # Some people ignore second dependency on the network weights  
  if constant_target:
    
    # The gradients for every weight matrix W_l in correct order, first element is None for zero-th input layer which
    # has no weights (only for convenience)
    # The executed action selects the row of the matrix-shaped gradient
    G_J_W = [None] + [TD * np.dot(psi[l].T, phi_block[l - 1])[_a] for l in xrange(1,L) ]

  else:

    # If not, the differential map with the primed input is also required     
    phi_prime_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi_prime)]

    # Same as before but with two terms resulting from the delta (reward is constant wrt. the weights and gets zero)
    G_J_W = [None] + [TD * (np.dot(psi[l].T, phi_block[l - 1])[_a] - gamma * np.dot(psi_prime[l].T, phi_prime_block[l - 1])[_a_prime]) for l in xrange(1,L)]

  #end

  # Now stack all flat gradients together to the full one. In this step also the first element is removed.
  # Due to the definition, this gradient is a lying row vector ( for the inner product with the vectorized direction )
  # -> This is a 1d array so transposed definition does not matter
  G_J_W_flat = np.concatenate(G_J_W[1:])

  return G_J_W_flat
def get_hessian_wrt_W_for_sample(params, phi, phi_prime, psi, psi_prime, gamma, constant_target, Q_DQN, a, a_prime):
  ''' 
  Returns the Hessian for the current (s,a,r,s',a')-tuple 
  
  Compare with eq. 27:
  - H_E (phi_L) is the Hessian of 1/2 delta^2 wrt. twice  the first, i.e. 1.0
  - Due to the different error expression (TD error in place of l2 norm) the calculations here are different 
    from those of the four region code
  - but the calculations here reuse parts of the Gradient, look at the comments above
  '''

  L = len(params)
  param_shapes = shapes(params[1:])
  
  if Q_DQN:
    _a = a
    _a_prime = a_prime
  else:
    _a = 0
    _a_prime = 0
  #end

  phi_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi)]

  if constant_target:
    phi_full = [ np.dot(psi[l].T, phi_block[l - 1])[_a] for l in xrange(1,L) ]
  else:
    phi_prime_block = [np.kron(np.eye(shape_k[1]), np.append(phi_k, 1.0).T) for shape_k, phi_k in izip(param_shapes, phi_prime)]
    phi_full = [ np.dot(psi[l].T, phi_block[l - 1])[_a] - gamma * np.dot(psi_prime[l].T, phi_prime_block[l - 1])[_a_prime] for l in xrange(1,L) ]
  #end

  # The single large block diagonal matrix
  phi_block_diag = sp.linalg.block_diag(*phi_full)

  # The final Hessian matrix
  H_J_W = np.dot(phi_block_diag.T, phi_block_diag)

  return H_J_W
def get_gradient_and_hessian_wrt_W_for_sample(params, non_linearities, s, a, r, s_prime, a_prime, gamma, no_hessian, constant_target, Q_factors, Q_DQN, **kwargs):
  ''' Returns the Gradient and Hessian wrt. W for the current (s,a,r,s',a')-tuple '''

  # The number of layers including the input layer   
  L = len(params)
    
  if Q_factors:
    
    # Collect the layer mappings for the tuples (s,a) and (s',a')
    # see eq. 3 & 4
    Q_sa = calculate_Q(s,a, params, non_linearities, False)
    Q_sa_prime = calculate_Q(s_prime, a_prime, params, non_linearities, False)
  
    # Collect the layer derivative mappings for the tuples (s,a) and (s',a')
    # see eq. 8 & 9
    Q_sa_deriv = calculate_Q(s,a, params, non_linearities, True)
    Q_sa_prime_deriv = calculate_Q(s_prime,a_prime, params, non_linearities, True)
    
    # Renaming to match notation in paper
    phi = Q_sa
    phi_prime = Q_sa_prime

    Sigma_prime = Q_sa_deriv
    Sigma_prime_s_prime = Q_sa_prime_deriv
  
  elif Q_DQN:
    
    # Here Q(s,a) corresponds to vector valued V(s) (handled by the arg parser argument "N"
    Q_sa = calculate_V(s, params, non_linearities, False)
    Q_sa_prime = calculate_V(s_prime, params, non_linearities, False)
  
    Q_sa_deriv = calculate_V(s, params, non_linearities, True)
    Q_sa_prime_deriv = calculate_V(s_prime, params, non_linearities, True)
    
    phi = Q_sa
    phi_prime = Q_sa_prime

    Sigma_prime = Q_sa_deriv
    Sigma_prime_s_prime = Q_sa_prime_deriv

  else:
    # Same as with Q_DQN, but with scalar V(s), code exists only for convenience
    V_s = calculate_V(s, params, non_linearities, False)
    V_s_prime = calculate_V(s_prime, params, non_linearities, False)

    V_s_deriv = calculate_V(s, params, non_linearities, True)
    V_s_prime_deriv = calculate_V(s_prime, params, non_linearities, True)
    
    phi = V_s
    phi_prime = V_s_prime

    Sigma_prime = V_s_deriv
    Sigma_prime_s_prime = V_s_prime_deriv

  #end

  # The recursive definition of psi_l (see eq 17). The last matrix Sigma_L has the index L-1 in the code. Be carful here!
  psi = [None] * (L - 1) + [np.diag(Sigma_prime[L - 1])]
  psi_prime = [None] * (L - 1) + [np.diag(Sigma_prime_s_prime[L - 1])]

  # Iterate backwards as in paper
  for l in xrange(L - 2,-1,-1):

    # In eq. 13 the direction h_l is applied to the layer input phi_l, thus the vector ends with an zero instead of one.
    # -> this indicates a direction in homogenous coordinates
    # -> the zero simply removes the last column in the transposed weight matrix (i.e. the bias vector)
    #    W.T[:,0:-1] == W[0:-1,:]
    rhs = np.dot(params[l + 1][0:-1,:], psi[l + 1])
    rhs_prime = np.dot(params[l + 1][0:-1,:], psi_prime[l + 1])

    # Then multiply with the next sigma to get psi_l
    psi[l] = np.dot(np.diag(Sigma_prime[l]) , rhs)
    psi_prime[l] = np.dot(np.diag(Sigma_prime_s_prime[l]) , rhs_prime)

  #end for

  G_J_W = get_gradient_wrt_W_for_sample(params, phi, phi_prime, psi, psi_prime, r, gamma, constant_target, Q_DQN, np.int(a), np.int(a_prime))
  
  if no_hessian:
    H_J_W = 0.0 # <- gets broadcasted correctly

  else:    
    H_J_W = get_hessian_wrt_W_for_sample(params, phi,phi_prime, psi, psi_prime, gamma, constant_target, Q_DQN, np.int(a), np.int(a_prime))

  return G_J_W, H_J_W

def build_gradient_and_hessian_wrt_W(params, non_linearities, data_wrapper, mu_a_given_s, pi_a_given_s, off_policy, **kwargs):
  ''' Calculates the complete gradient and hessian, i.e. performs the sum over all samples (eq. 36) '''
  
  N_net = count_parameter(params)

  H_W = np.zeros((N_net, N_net))
  G_W = np.zeros((N_net,))
  
  T = len(data_wrapper)

  for i in xrange(T):

    s,a,r,s_prime, a_prime = data_wrapper[i]
    gamma = data_wrapper.gamma

    G,H = get_gradient_and_hessian_wrt_W_for_sample(params, non_linearities,s,a,r,s_prime, a_prime, gamma, **kwargs)

    # Importance sampling weights have to be included in front of the (s,a,r,s',a')-tuple
    if off_policy:
      rho = pi_a_given_s(s,a) / mu_a_given_s(s,a)
    else:
      rho = 1.0
    #end if

    G_W += rho * G
    H_W += rho * H

  #end for
  
  # New for RL: An expectation is approximated with samples, thus we need the mean here
  G_W /= T
  H_W /= T

  return G_W, H_W
def descent_step(params, non_linearities, data_wrapper, mu_a_given_s, pi_a_given_s, alpha, delta, no_hessian, **kwargs):
  ''' A single training step '''

  W_shapes = shapes(params[1:])

  G_W, H_W = build_gradient_and_hessian_wrt_W(params, non_linearities, data_wrapper, mu_a_given_s, pi_a_given_s, no_hessian = no_hessian, **kwargs)

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
  error_new = calculate_error(params_new, non_linearities, data_wrapper, **kwargs)

  return params_new, error_new      

def train(params, non_linearities, data_wrapper, mu, mu_a_given_s, pi_a_given_s, run, iterations, plotting_rate, semi_online, show_progress, **kwargs):
  ''' The main training loop for an agent, returns the lists with parameters and errors over time and the required time '''

  t0 = time.time()
  
  errors_all = []
  params_all = []

  # Before the semi online training starts the wrapper has to be in a fresh state.
  if semi_online is True:
    data_wrapper.reset()

  it = xrange(iterations)

  if show_progress:
    it = progbar()(it)
    
  for i in it:

    # For semi online training after every step new data has to be collected, starting in the last state of the last round
    if semi_online is True:
      data_wrapper.create_data_set(policy = mu, reset = False)

    # A single gradient descent step
    params, error = descent_step(params, non_linearities, data_wrapper, mu_a_given_s, pi_a_given_s, **kwargs)
    
    errors_all.append(error)
    params_all.append(params)
    
    if plotting_rate > 0 and (i + 1) % plotting_rate == 0:
      plot_error(run, errors_all)
      #plot_weight_distance(run, params_all)
      plt.show()
    #end if

    how_many = 10
    err_mean = np.array(errors_all[-how_many:]).mean()

    threshold = 1e-15
    if err_mean < threshold:
      logger.info("Mean error of the last {} steps smaller than {} after {} iterations, terminating.".format(how_many, threshold, i))
      break
    #end

    threshold = 1e150
    if err_mean > threshold:
      logger.info("Mean error of the last {} steps larger than {} after {} iterations, terminating.".format(how_many, threshold, i))
      break
    #end

  #end for

  t1 = time.time()

  return params_all, errors_all, t1 - t0
def policy_iteration(T, env, Q_DQN, **kwargs):
  ''' Runs policy iteration '''

  errors_star_all = []
  params_star_all = []
  reward_all = []
  length_all = []

  # TODO: Add convergence check
  policy_converged = False
  
  # Counts the number of Policy Iteration steps
  counter = 0

  # First data comes from random policy
  data_wrapper = get_data_wrapper(env, T = T)

  # Initial Q-function is arbitary
  params_star, non_linearities = create_network(**kwargs)

  while not policy_converged and counter < 200:

    # Training corresponds to policy evaluation    
    params_all, errors_all, timing = train(params_star, non_linearities, data_wrapper, None, None,None,Q_DQN = Q_DQN, **kwargs)

    # "New" Q-function -> last parameter vector
    params_star = params_all[-1]
    params_star_all.append(params_star)

    # The final error of the "New" Q-function
    errors_star = errors_all[-1]
    errors_star_all.append(errors_star)

    # Calculate for the given state s the expected reward Q_pi(s,a) for all actions a,
    # return the action corresponding to the largest Q-factor.
    if Q_DQN:
      policy = lambda s: calculate_V(s,params_star, non_linearities, False)[-1].argmax()
    else:
      policy = lambda s: np.array([ calculate_Q(s,a, params_star, non_linearities, False)[-1] for a in xrange(data_wrapper.a_dim)]).argmax()
    #end

    eps_greedy_policy = lambda s: policy(s) if np.random.rand(0,1) < 0.5 else data_wrapper.env.action_space.sample()

    # Show the policy
    data_wrapper.test_policy(policy, 1, render = True)
    # and test actual performance by NOT using the eps-greedy policy
    reward, length = data_wrapper.test_policy(policy, 10, render = False)

    reward_all.append(reward)
    length_all.append(length)

    # Collecting new data with the current policy induced by the Q-function defines the policy improvement step
    data_wrapper.create_data_set(eps_greedy_policy)

    msg = '''PI step {:3d}:
    reward: {}
    length: (mean {}|min {}|max {})
    error: {}'''

    logger.info(msg.format(counter, reward[0], length.mean(), length.min(), length.max(), errors_star))

    counter += 1
  #end

  return params_star_all, errors_star_all, reward_all, length_all

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# These functions plot some stuff
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_error(run, errors):
  fig = plt.figure()
  fig.canvas.set_window_title(run)
  plt.plot(errors, "-b")
  plt.xlabel("iterations")
  plt.ylabel("error")
  plt.title("(Test set) error over time")
  plt.yscale("log")
  plt.grid()
  return
def plot_weight_distance(run, weights):
  ''' 
  Plots the distance from the last weight in the list to show the convergence speed. The
  distance is meassured by the l2 norm
  '''
  
  W_distance = calculate_weight_distance(weights)

  fig = plt.figure()
  fig.canvas.set_window_title(run)
  plt.plot(W_distance, "-b")
  plt.xlabel("iterations")
  plt.ylabel("distance")
  plt.title("distance from ground truth")
  plt.yscale("log")
  plt.grid()

  return
def plot_MC_value_functions(args):
  Q_MC, V_MC = task_MC_PE(args)
  
  # Required for its plotting functions  
  data_wrapper = get_data_wrapper("MyMountainCar-v0", use_file = True, filename = file_mountaincar_data_fixed_policy)
  data_wrapper.plot_V_function(V_MC, "V_MC")
  
  plt.tight_layout()
  filename_pdf = os.path.join(".","graphics", "V_MC.pdf")
  plt.savefig(filename_pdf,bbox_inches='tight', pad_inches=0)
def plot_value_function(results, args, clip=False):
    
  V_net, params, errors, timing = results

  # Only required for the plotting functions
  data_wrapper = get_data_wrapper("MyMountainCar-v0",use_file = True, filename = file_mountaincar_data_fixed_policy)

  filename = "V_net_{}".format(args.run)

  if clip:
    V_net = V_net.clip(-11,-8)

  data_wrapper.plot_V_function(V_net, filename)
  
  plt.tight_layout()
  filename_pdf = os.path.join(".","graphics", "{}.pdf".format(filename))
  plt.savefig(filename_pdf,bbox_inches='tight', pad_inches=0)
  
  return

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# These functions represent the tasks that can be started
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def task_dummy(args, **kwargs):
  ''' A dummy task, which is the dault value for the arg parser '''
  return
def task_PI(args, **kwargs):

  filename = "PI_results_{}.pickle".format(args.run)

  results = the_time_saver(policy_iteration, os.path.join(build_dir,filename), args.recreate,
                           **vars(args))
  return results
def task_MC_PE(args, **kwargs):
  ''' This task creates ground truth solution via Monte Carlo methods '''

  data_wrapper = get_data_wrapper("MyMountainCar-v0",T = 10)
  
  local_build_dir = kwargs.get("build_dir", os.path.join(".","build"))
  
  file_Q_MC = "Q_MC_moutaincar_fixed_policy.pickle"
  Q_MC = the_time_saver(data_wrapper.monte_carlo_Q_function, os.path.join(local_build_dir, file_Q_MC), args.recreate,
                        MountainCarWrapper.fixed_policy, runs_per_start = 15)

  file_V_MC = "V_MC_moutaincar_fixed_policy.pickle"
  V_MC = the_time_saver(data_wrapper.monte_carlo_V_function, os.path.join(local_build_dir,file_V_MC), args.recreate,
                        MountainCarWrapper.fixed_policy, runs_per_start = 15)

  return Q_MC, V_MC
def task_NN_PE(args, **kwargs):
  ''' 
  This task trains a feed forward network to approximate the value function for the fixed policy
  Thus this task always requires the MountainCar world
  '''
  
  if args.random_data:
    data_wrapper = get_data_wrapper(args.env, T = args.T, policy = MountainCarWrapper.fixed_policy)
  else:                             
    data_wrapper = get_data_wrapper(args.env, use_file = True, filename = file_mountaincar_data_fixed_policy)
  #end

  params_init, non_linearities = create_network(**vars(args))

  # This choice restores the old task, mu is executed in the case of semi online to get the data.
  # The importance weights are not used
  mu = MountainCarWrapper.fixed_policy
  mu_a_given_s = None
  pi_a_given_s = None
  
  filename = "V_net_{}.pickle".format(args.run)
  params, errors, timing = the_time_saver(train, os.path.join(build_dir, filename), args.recreate,
                                          params_init, non_linearities, data_wrapper,
                                          mu, pi_a_given_s, mu_a_given_s, **vars(args))

  params_star = params[-1]
  
  filename = "V_net_eval_{}.pickle".format(args.run)
  V_net = the_time_saver(evaluate_V, os.path.join(build_dir, filename), args.recreate,
                         params_star, non_linearities, data_wrapper)

  return V_net, params, errors, timing
def task_off_pol_MC_PE(args, **kwargs):
  ''' 
  This task creates ground truth solution for off policy learning
  by using Monte Carlo methods. Make shure the target_policy is set up correctly! 
  '''

  data_wrapper = get_data_wrapper("MyMountainCar-v0", T = 10)
  
  local_build_dir = kwargs.get("build_dir", os.path.join(".","build"))

  mu_prop_correct_a = 0.4
  pi_prop_correct_a = 0.8

  mu = lambda obs : MountainCarWrapper.noisy_fixed_policy(obs, mu_prop_correct_a)
  pi = lambda obs : MountainCarWrapper.noisy_fixed_policy(obs, pi_prop_correct_a)

  mu_a_given_s = lambda s,a: MountainCarWrapper.p_for_choice(MountainCarWrapper.fixed_policy(s), 3, mu_prop_correct_a)[np.int(a)]
  pi_a_given_s = lambda s,a: MountainCarWrapper.p_for_choice(MountainCarWrapper.fixed_policy(s), 3, pi_prop_correct_a)[np.int(a)]

  filename = "V_MC_80_percent.pickle"
  V_MC = the_time_saver(data_wrapper.monte_carlo_V_function, os.path.join(local_build_dir,filename), args.recreate,
                        pi, runs_per_start = 15)

  return V_MC
def task_off_pol_NN_PE(args, **kwargs):
  ''' 
  This task evaluates the target policy (e.g. the fixed policy of the MoutainCarWrapper) 
  Thus this task always requires the MountainCar world
  '''
  
  if args.random_data:
    data_wrapper = get_data_wrapper(args.env, T = args.T)
  else:                             
    data_wrapper = get_data_wrapper(args.env, use_file = True, filename = file_mountaincar_data_off_pol_pe)
  #end

  params_init, non_linearities = create_network(**vars(args))

  mu_prop_correct_a = 0.4
  pi_prop_correct_a = 0.8

  mu = lambda obs : MountainCarWrapper.noisy_fixed_policy(obs, mu_prop_correct_a)
  pi = lambda obs : MountainCarWrapper.noisy_fixed_policy(obs, pi_prop_correct_a)

  mu_a_given_s = lambda s,a: MountainCarWrapper.p_for_choice(MountainCarWrapper.fixed_policy(s), 3, mu_prop_correct_a)[np.int(a)]
  pi_a_given_s = lambda s,a: MountainCarWrapper.p_for_choice(MountainCarWrapper.fixed_policy(s), 3, pi_prop_correct_a)[np.int(a)]

  # Run the off policy learning
  filename = "V_net_{}.pickle".format(args.run)
  params, errors, timing = the_time_saver(train, os.path.join(build_dir, filename), args.recreate,
                                          params_init, non_linearities, data_wrapper,
                                          mu, mu_a_given_s, pi_a_given_s, **vars(args))

  # Train the value function in the same way but using Monte Carlo methods
  # -> Ground truth
  # -> Takes quite some time, the result is provided as pickled file
  V_MC = task_off_pol_MC_PE(args)

  # Now calculate the distance between the MC value function and the value function defined by the parameters over time
  filename = "V_net_dist_{}.pickle".format(args.run)
  V_net_dist = the_time_saver(evaluate_distance_V_and_V_MC, os.path.join(build_dir, filename), args.recreate,
                              V_MC, params, non_linearities, data_wrapper, show_progress = args.show_progress)


  fig = plt.figure()
  fig.canvas.set_window_title(args.run)
  plt.plot(V_net_dist, "-b")
  plt.xlabel("iterations")
  plt.ylabel("distance")
  plt.title("distance from ground truth")
  plt.yscale("log")
  plt.grid()
  plt.show()
  
  # Return in the correct order the elapsed time
  return None, None, None, timing 
def task_test_error(args, **kwargs):
  ''' Evaluates the final network weights after the training is over with a dedicated test data set '''
  data_wrapper = get_data_wrapper("MyMountainCar-v0", T = args.T, policy = MountainCarWrapper.fixed_policy)

  filename = "V_net_{}.pickle".format(args.run)
  params, errors, timing = the_time_saver(dummy, os.path.join(build_dir, filename), make_new = False)

  params_star = params[-1]
  _, non_linearities = create_network(**vars(args))

  filename = "V_test_error_{}.pickle".format(args.run)
  test_error = the_time_saver(calculate_error, os.path.join(build_dir, filename), args.recreate,
                              params_star, non_linearities, data_wrapper, **vars(args))

  return test_error
def task_test_error_full(args, **kwargs):
  ''' Similar to task_test_error, but returns the test error for all iterations in the training phase '''

  data_wrapper = get_data_wrapper("MyMountainCar-v0", T = args.T, policy = MountainCarWrapper.fixed_policy)

  filename = "V_net_{}.pickle".format(args.run)
  params, errors, timing = the_time_saver(dummy, os.path.join(build_dir, filename), make_new = False)

  _, non_linearities = create_network(**vars(args))

  def do_it():
    test_error_full = np.empty((len(params)))
    for i, param in enumerate(params):
      test_error_full[i] = calculate_error(param, non_linearities, data_wrapper, **vars(args))
    return test_error_full

  filename = "V_test_error_full_{}.pickle".format(args.run)
  test_error_full = the_time_saver(do_it, os.path.join(build_dir, filename), args.recreate)

  return test_error_full

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code to run the stuff
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def parse_arguments(*args, **kwargs):
  ''' Creates the parser and returns the parsed result '''
  import argparse
  parser = argparse.ArgumentParser(description = "Gauss Newton for Deep Reinforcement Learning")
  
  parser = parser_add_common_arguments(parser)

  task_names = [ k for k in globals().keys() if k.startswith("task_")]
  parser.add_argument('-t', '--task', type = str, choices = task_names, default = "task_dummy",
                      help = "The task to run, i.e. one of the task function names. Default is %(default)s.")
  
  #get_class_name = lambda c: str(c).split('.')[-1][0:-2 - 7]#-7 to remove "Wrapper"
  #env_names = map(get_class_name, [MountainCarWrapper, CartPoleWrapper])
  env_names = [foo.id for foo in gym.envs.registry.all()]
  
  parser.add_argument('-e', '--env', type = str,choices = env_names, default = 'MyMountainCar-v0',
                      help = "The environment for the agent. Default is %(default)s.")

  parser.add_argument('-so', '--semi_online', action="store_true",
                      help = "If specified, a randomized roll-out of the specified length is performed after each descent step. Together with -T 1 this"
                      " training method becomes 'true' online. If -so is not specified the data is collected at the beginning and used for the complete training.")
  parser.add_argument('-ct', '--constant_target', action="store_true",
                      help = "If specified, the TD-target is treated as constant and is not considered in the differential map.")  
  parser.add_argument('-Q', '--Q_factors', action="store_true",
                      help = "If specified, Q-factors are used.")  
  parser.add_argument('-Q2', '--Q_DQN', action="store_true",
                      help = "If specified, a DQN-like network layout is used.")
  parser.add_argument('-op', '--off_policy', action="store_true",
                      help = "If specified, importance sampling weights rho = pi(a|s) / mu(a|s) are calculated "
                      "and used in front of the squared TD error.")
  
  args,unkowns = parser.parse_known_args(*args)

  if len(unkowns) != 0:
    logger.warn("Unrecognized arguments are present: {}".format(unkowns))

  # Some checks on parameter combinations
  if args.semi_online:
    assert args.random_data, "The environment wrapper is not allowed to use data from a file in the semi online mode"

  if args.task == 'task_PI':
    assert args.Q_factors or args.Q_DQN, "The policy iteration task requires some form of Q-factors"

  if args.task == 'task_NN_PE':
    assert args.env == 'MyMountainCar-v0', "Neural Policy Evaluation requires the fixed policy in the MountainCar environment"
    assert not args.off_policy, "The task Neural Policy Evaluation has no rho defined."

  if args.task == 'task_off_pol_NN_PE':
    assert args.env == 'MyMountainCar-v0', "Off Policy Neural Policy Evaluation requires the fixed policy in the MountainCar environment"
    assert args.off_policy, "For off policy policy evaluation the importance sampling weights are required."

  parser_finalize(args,**kwargs)

  return args

if __name__ == "__main__":
    
  # Uncomment one of the following lines and let the code run
  # This is actually the same as using the arg parser from the command line
  # For a description please use the help of the arg parser

  #--------------------------------------
  # Demonstration of Policy Iteration like algorithm 
  #--------------------------------------

  # Uses the CartPole environment (not the original one, but the one from this repo with the adjusted reward)
  # A Deep Q-Network like architecture is used.
  # Balancing should occur after 20 PI iterations
  args = parse_arguments( "run_1 -t task_PI -e MyCartPole-v0 -d 1e-5 --Q_DQN -n 4 10 10 2 -rw -a 0.01 -T 300 -i 15 -sp".split())
    
  # Similar to run_1, but with a different network layout. State and action are combined and fed to the network
  # thus 4 + 1 input units and 1 output
  #args = parse_arguments( "run_2 -t task_PI -e MyCartPole-v0 -d 1e-5 -Q -n 5 10 10 1 -rw -a 0.01 -T 300 -i 15 -sp".split())
    
  #--------------------------------------
  # Demonstration of off-policy learning
  #--------------------------------------

  # Gauss Newton with out constant TD-target  
  #args = parse_arguments("run_3 -t task_off_pol_NN_PE -n 2 10 10 1 -i 5000 -d 1e-5 -so -op -rd -rw -a 0.001 -T 10 -fw init_weights_{run}_{n}.pickle -sp".split())

  # Gauss Newton with constant TD-target
  #args = parse_arguments("run_4 -t task_off_pol_NN_PE -n 2 10 10 1 -i 5000 -d 1e-5 -so -op -rd -rw -a 0.001 -T 10 -fw init_weights_{run}_{n}.pickle -sp -ct ".split())

  # Gradient only with out constant TD-target
  #args = parse_arguments("run_5 -t task_off_pol_NN_PE -n 2 10 10 1 -i 5000 -d 1e-5 -so -op -rd -rw -a 0.001 -T 10 -fw init_weights_{run}_{n}.pickle -sp -nh".split())

  # Gradient only with constant TD-target
  #args = parse_arguments("run_6 -t task_off_pol_NN_PE -n 2 10 10 1 -i 5000 -d 1e-5 -so -op -rd -rw -a 0.001 -T 10 -fw init_weights_{run}_{n}.pickle -sp -nh -ct".split())
  
  # If argument have been specified in the command line the hard coded ones are overwritten
  if len(sys.argv) > 1:
    args = parse_arguments()

  logging.info("Launching task '{}' in environment '{}'".format(args.task, args.env))

  tasks = { k:v for k,v in locals().iteritems() if k.startswith("task_")}
  results = tasks[args.task](args)
  
  V_net, params, errors, timing = results

  print "Elapsed time: ", timing
   