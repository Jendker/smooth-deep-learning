import logging
logger = logging.getLogger(__name__)

import numpy as np

import cPickle
import os
from progressbar import ProgressBar as progbar

import fixed_gym_envs
import gym

class EnvironmentWrapper(object):
  ''' 
  A simple wrapper for the open ai gym environments to perform roll-outs, evaluate policies, etc.
  '''
  def __init__(self, env_name, T=1000, use_file=False, filename=None, policy=None, *args, **kwargs):
    ''' 
    Creates an Environment Wrapper instance. 
    
    If use_file is true, the wrapper tries to restore the (s,a,r,s',a') tuples from a file instead of
    creating a roll-out with the specified policy (random if None is given).
    Using a file also overwrites the specified trajectory length.
    '''
    
    self.env = gym.make(env_name)
    
    # Only for storing the pickle files
    self.build_dir = os.path.join(".", kwargs.get("build_dir", "build"))

    # The dimension of the state space and the number of discrete actions
    self.o_dim = self.env.observation_space.shape[0]
    self.a_dim = self.env.action_space.n

    # The length of a trajectory (or roll-out)
    self.T = T

    # Variables for converting discrete states to continuous observations
    self.o_high = self.env.observation_space.high 
    self.o_low = self.env.observation_space.low 
    self.o_range = zip(self.o_low,self.o_high)
    
    # Storing states and states primes wastes memory, but makes the remaining code easier
    # (no extra effort to create valid (s,a,r,s',a') tuples while resetting at end of episode)
    self.states = None
    self.states_prime = None

    # Discrete action space -> only 1dim array to store the integer actions    
    self.actions = None
    self.actions_prime = None
        
    # Rewards are by definition a scalar -> 1d array
    self.rewards = None

    # The discount factor, currently hardcoded
    self.gamma = 0.9

    # This variable stores the last state of an trajectory. Thereby the state of the world can be 
    # maintained accross different roll-outs, which is required for online learning
    self.s = None

    if use_file is True:

      if filename is None:
        raise ValueError("A filename is required")
      
      self.load_data_Set(filename)

    else:

      # Fill the initial data with random policy
      self.create_data_set(policy)

    #end

    return super(EnvironmentWrapper, self).__init__(*args, **kwargs)
  def __len__(self): 
    ''' Retrurns the number of stored (s,a,r,s',a') tuples '''
    return self.T
  def __getitem__(self, i):
    ''' Returns the i-th (s,a,r,s',a') tuple '''
    return self.states[i], self.actions[i], self.rewards[i], self.states_prime[i], self.actions_prime[i]

  def create_data_set(self, policy=None, **kwargs):
    '''
    Call this function with a policy to create (s,a,r,s',a') tuples 
   
    Default policy is random exploration provided by the underlying openai gym environment.

    The keywords arguments are forwarded to the roll_out method, look at its doc string
    for further details about possible arguments.
    '''

    self.states, self.actions, self.rewards, self.states_prime, self.actions_prime, _ = \
      self.roll_out(policy = policy, trajectory_length = self.T, **kwargs)

    return 
  def save_data_set(self, filename):
    ''' Dumps the current (s,a,r,s',a') tuple into a file '''

    if not os.path.exists(self.build_dir):
      os.mkdir(self.build_dir)
    #end if
    
    with open(os.path.join(self.build_dir, filename), "wb") as the_file:      
      cPickle.dump((self.states, self.actions, self.rewards, self.states_prime, self.actions_prime),
                   the_file, cPickle.HIGHEST_PROTOCOL)
    #end with

    return
  def load_data_Set(self, filename):
    ''' Restores the (s,a,r,s',a') tuples from the specified file '''
    
    with open(os.path.join(self.build_dir, filename), "rb") as the_file:
      result = cPickle.load(the_file)
    #end with

    self.states, self.actions, self.rewards, self.states_prime, self.actions_prime = result

    assert all([result[0].shape[0] == foo.shape[0] for foo in result]), "Data arrays with differnt lengths detected."
    
    self.T = self.states.shape[0]

    logger.info("States, actions, rewards etc. loaded from file.")
    return

  def roll_out(self,policy=None, start_state=None, start_action=None, trajectory_length=300, render=False, reset=True):
    '''
    Creates a trajectory of the form s_0,a_0,r_0,s_1,a_1,... using the provided policy.
    
    The trajectory starts with the given (s,a) tuple, i.e. s_0 = s and a_0 = a, if one is given.

    The trajectory length determines how many steps are performed in the world.

    If the argument 'render' is set to True the roll-outs are shown. This has a significant impact on the performance.

    If 'reset' is set to false and no start state or action is given, the trajectory starts at the last 
    state of the previous call to this function.
    Thereby an online learning becomes possible. True online data is obtained by setting additionally the 
    length of the trajectory to T=1.
    '''

    if policy is None:
      policy = self.random_policy
      
    states = np.empty((trajectory_length, self.o_dim))
    states_prime = np.empty_like(states)
    actions = np.empty((trajectory_length,))
    actions_prime = np.empty_like(actions)
    rewards = np.empty((trajectory_length,))

    if reset is True:# or self.s is None:
      self.s = self.env.reset()
    #end

    # Overwrite reset observation if desired
    if start_state is not None:
      self.s = start_state
      self.env.state = start_state

    episode_lengths = []
    steps = 0

    # Create the actual roll-out
    for t in xrange(trajectory_length):

      if render: 
        self.env.render()
      #end if

      # Set action in first time step to given start action
      a = start_action if t == 0 and start_action is not None else policy(self.s)
      
      # Depending on the world the reward of the openai env is ignored
      s_prime, r, done, _ = self.env.step(a)
      
      # Teleport back into starting region -> ergodic problem is obtained
      if done:
        s_prime = self.env.reset()

        episode_lengths.append(steps)
        steps = 0
      #end
    
      a_prime = policy(s_prime)

      # Store SARSA tuple
      states[t,:] = self.s
      states_prime[t,:] = s_prime
      actions[t] = a
      actions_prime[t] = a_prime
      rewards[t] = r
    
      # Prepare next round
      self.s = s_prime
      steps +=1

    #end for t

    if len(episode_lengths) == 0:
      episode_lengths.append(steps)

    return states, actions, rewards, states_prime, actions_prime, episode_lengths
  def test_policy(self, policy, repeats=15, render=False):
    '''
    Call this function with a policy to evaluate its performance by performing several roll-outs.

    The function returns the discounted rewards for all roll-outs and the lengths of all episodes
    (i.e. the number of steps until the transition from a goal state to one of the start states occurs)
   
    Set 'render' to true if you want to see the evaluation
    '''

    rewards = np.empty((repeats,))
    all_episode_lengths = []
    
    for i in xrange(repeats):
    
      _,_,reward,_,_,steps = self.roll_out(policy = policy, render = render)
      rewards[i] = EnvironmentWrapper.apply_discount(reward, self.gamma)
      all_episode_lengths += steps

    #end

    return rewards, np.array(all_episode_lengths)
  def expected_return(self,o, a=None, runs_per_start=10):
    ''' Runs several trajectories and averages them to get the expected return. '''
          
    reward_rollouts = np.empty((runs_per_start,))

    for t in xrange(runs_per_start):
      _,_,rewards,_,_,_ = self.roll_out(o, a)
      reward_rollouts[t] = EnvironmentWrapper.apply_discount(rewards, self.gamma)
    #end for
      
    return reward_rollouts.mean()

  def random_policy(self,s):
    return self.env.action_space.sample()

  def reset(self):
    ''' Resets the internal world, should be called before starting online training '''
    self.s = self.env.reset()

  @staticmethod
  def apply_discount(costs, discount):
    return np.sum([ np.power(discount, i) * r for i,r in enumerate(costs) ])

class MountainCarWrapper(EnvironmentWrapper):
  '''
  A subclass for the Environment Wrapper to provide specialized plotting functions and
  iterators over the state action space for the Monte Carlo methods
  '''
  def __init__(self, T=1000, use_file=False, filename=None, policy=None, *args, **kwargs):

    # For discretizing the 2D Statespace
    self.binsX = 100
    self.binsY = 100

    return super(MountainCarWrapper, self).__init__("MyMountainCar-v0", T, use_file, filename, policy, *args, **kwargs)

  @staticmethod
  def fixed_policy(obs):
    ''' Hand crafted optimal policy '''
    pos, vel = obs

    # Left
    if vel <= 0.0: return 0 

    # Right
    if vel > 0.0: return 2
    
    #Else no acceleration
    return 1
  @staticmethod
  def noisy_fixed_policy(obs, p=0.8):
    ''' Returns with probability p the output of the fixed policy and with (1-p)/2 one of the two others '''
    a = MountainCarWrapper.fixed_policy(obs)
    all_p = MountainCarWrapper.p_for_choice(a,  3, p)
    return np.random.choice([0,1,2], p = all_p)

  @staticmethod
  def p_for_choice(idx, n, p):
    ''' 
    Returns the probabilities for the n entries of a list to be picked.
    The desired element given by idx gets p, the n-1 others get (1-p)/(n-1)
    '''
    all_p = [(1.0 - p) / (n - 1)] * n
    all_p[idx] = p
    return all_p
  
  def monte_carlo_Q_function(self, policy, runs_per_start=10, show_progress=True):
    '''
    Calcuates the Q-function under the given policy for the mountaincar environment 
    via Monte Carlo roll-outs.
    
    This functions sets via loops the start state and action and runs several times 
    sufficiently long trajecotries with the policy.

    The expected reward for the (s,a) tuples is obtained via averaging the rewards of all roll-outs.
    '''

    logger.info("Creating MC Q-Function")

    Q_MC = np.empty((self.binsX, self.binsY, self.a_dim))
       
    iter = self.iterate_state_action_space()

    if show_progress:
      iter = progbar()(iter, max_value = Q_MC.size)

    for (i,j,k), (o,a) in iter:

      reward_rollouts = np.empty((runs_per_start,))

      for t in xrange(runs_per_start):
        _,_,rewards,_,_,_ = self.roll_out(policy,o, a)
        reward_rollouts[t] = EnvironmentWrapper.apply_discount(rewards, self.gamma)
      #end for
      
      Q_MC[i,j,k] = reward_rollouts.mean()

    #end for

    return Q_MC
  def monte_carlo_V_function(self, policy, runs_per_start=10, show_progress=True):
    '''
    Calcuates the value-function under the given policy for the mountaincar environment 
    via Monte Carlo roll-outs.
    
    This functions sets via a loop the start state runs several times sufficiently long 
    trajecotries with the policy.

    The expected reward for each starting state is obtained via averaging the rewards of all roll-outs.
    '''

    logger.info("Creating MC value-function")

    V_MC = np.empty((self.binsX, self.binsY))
   
    iter = self.iterate_state_space()

    if show_progress:
      iter = progbar()(iter, max_value = V_MC.size)

    for (i,j), o in iter:

      reward_rollouts = np.empty((runs_per_start,))

      for t in xrange(runs_per_start):
        _,_,rewards,_,_,_ = self.roll_out(policy, o)
        reward_rollouts[t] = EnvironmentWrapper.apply_discount(rewards, self.gamma)
      #end for
      
      V_MC[i,j] = reward_rollouts.mean()

    #end for

    return V_MC

  def plot_V_function(self, V, window_title="", z_label="-V"):
    ''' 
    Plots the Value function for the mountaincar world 
    
    The value function is assumed to be a matrix with shape (binsX, binsY)

    The Z-axis is inverted (so negative rewars point upwards)
    '''
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    assert V.shape == (self.binsX, self.binsY), "For plotting the Value function has to be a matrix"

    x = np.linspace(self.o_low[0], self.o_high[0], self.binsX)
    y = np.linspace(self.o_low[1], self.o_high[1], self.binsY)
    X,Y = np.meshgrid(x,y)
  
    fig = plt.figure()
    fig.canvas.set_window_title(window_title)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y, -V , cmap = 'RdBu', rstride=4, cstride=4,alpha=1.0) 

    #cset = ax.contour(X, Y, -V, zdir='z', offset=0, cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, -V, zdir='x', offset=param['binsY'], cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, -V, zdir='y', offset=param['binsX'], cmap=cm.coolwarm)
  
    ax.set_xlabel('position')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax.set_ylabel('velocity')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  
    ax.set_zlabel(r'${}$'.format(z_label))
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))

    # Rotate in x-y plane
    ax.view_init(azim = 45)
    
    # Viewing distance to avoid cutting off labels in pdf
    ax.dist = 11

    return
  def plot_V_function_contour(self, V, title):
    ''' 
    Plots the Value function for the mountaincar world 
    
    The value function is assumed to be a matrix with shape (binsX, binsY)

    The Z-axis is inverted (so negative rewars point upwards)
    '''
    
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    #import scipy.ndimage

    assert V.shape == (self.binsX, self.binsY), "For plotting the Value function has to be a matrix"

    extent = (self.o_low[0], self.o_high[0], self.o_low[1], self.o_high[1])
  
    fig = plt.figure()


    im = plt.imshow(V, interpolation = "gaussian", cmap = "OrRd", origin = "lower", extent = extent, aspect="auto")
    v = plt.axis()

    #V_smooth = scipy.ndimage.zoom(V, 2)
    CS = plt.contour(V, origin = "lower", extent = extent)
    plt.clabel(CS, inline = 1, fontsize = 8)

    plt.axis(v)

    plt.colorbar(im)
    
    plt.xlabel('position')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
   
    plt.ylabel('velocity')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  
    plt.title(r'${}$'.format(title))

    return

  def iterate_state_space(self):
    ''' 
    Can be used in a for loop, yields a discrete state index and the corresponding obseravtion.
    The index can be used as index for the V-function in tabular form, the observation can be fed into a neural
    network.
    '''
    for i,x in enumerate(np.linspace(self.o_low[0], self.o_high[0], self.binsX)):
      for j,y in enumerate(np.linspace(self.o_low[1], self.o_high[1], self.binsY)):
          o = np.array([x,y])
          yield (i,j),o

    raise StopIteration
  def iterate_state_action_space(self):
    ''' 
    Can be used in a for loop, yields a discrete state index and the corresponding obseravtion and action.
    The index can be used as index for the Q-function in tabular form, the observation can be fed into a neural
    network.    
    '''

    for (i,j),o in self.iterate_state_space():
      for a in xrange(self.a_dim):
        yield (i,j,a),(o,a)
    
    raise StopIteration

def get_data_wrapper(env, *args, **kwargs):
  '''A little helper to specify the behavior for individual worlds'''

  if env == 'MyMountainCar-v0':
    return MountainCarWrapper(*args, **kwargs)
  else:  
    return EnvironmentWrapper(env, *args, **kwargs)

if __name__ == "__main__":

  wrapper = get_data_wrapper("MyCartPole-v0",T = 100)
  wrapper.roll_out(trajectory_length = 1000, render = True, reset = False)

  print "All done"
