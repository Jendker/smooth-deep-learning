"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import numpy as np

import gym
import gym.spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)

class ContinuousCartPoleEnv(gym.Env):

  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50}

  def __init__(self):
    self.gravity = 9.8
    self.masscart = 1.0
    self.masspole = 0.1
    self.total_mass = (self.masspole + self.masscart)
    self.length = 0.5 # actually half the pole's length
    self.polemass_length = (self.masspole * self.length)
    self.force_mag = 10.0
    self.tau = 0.02 # seconds between state updates

    # Angle at which to fail the episode
    self.theta_threshold_radians = 12 * 2 * np.pi / 360.0
    self.x_threshold = 2.4 

    # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
    # Velocities were inf before, now capped at 10    
    high = np.array([self.x_threshold * 2,
      10.0,
      self.theta_threshold_radians * 2,
      10.0])
    self.observation_space = gym.spaces.Box(-high, high, dtype = np.float64)
    
    # Min and max acceleration for cart
    action_border = np.array([1.0])    
    self.action_space = gym.spaces.Box(-action_border, action_border, dtype = np.float64)

    self.seed()
    self.reset()
    self.viewer = None

    # Debugging variables for my agent
    # self.debug_pos_cart = 0.0
    # self.debug_incremet = 0.1

    self.step_counter = 0

    # Due to inconsitencies in the TimeWrapper we cannot rely on this one
    # Thus we introduce a custom limit in the world which terminates an episode
    self.max_episode_steps = 300
    
  def _base_step(self, _action):

    self.step_counter += 1

    x, x_dot, theta, theta_dot = self.state

    # Extract the scalar float action, remaining calculations have correct shape
    action = _action[0]
    
    force = self.force_mag * action

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
    thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
    xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
    x = x + self.tau * x_dot
    x_dot = x_dot + self.tau * xacc
    theta = theta + self.tau * theta_dot
    theta_dot = theta_dot + self.tau * thetaacc

    self.state = (x,x_dot,theta,theta_dot)

    time_is_up = self.step_counter >= self.max_episode_steps

    beyond_borders = not -self.x_threshold < x < self.x_threshold or \
                     not -self.theta_threshold_radians < theta < self.theta_threshold_radians

    done = beyond_borders or time_is_up
    
    reward = -1.0 if beyond_borders else 0.0

    return np.array(self.state), reward, done, {}
  
  # Implementation of interface

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, _action):
   
    assert self.action_space.contains(_action), "%r (%s) invalid" % (_action, type(_action))
    
    return self._base_step(_action)

  def reset(self):
    self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

    self.step_counter = 0

    # Random state disabled for now to debug
    # self.state = np.array([0.0]*4)

    # return np.array(self.state) # Crap line removed
    return self.state

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400

    world_width = self.x_threshold * 2
    scale = screen_width / world_width
    carty = 100 # TOP OF CART
    polewidth = 10.0
    polelen = scale * 1.0
    cartwidth = 50.0
    cartheight = 30.0

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      l,r,t,b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
      axleoffset = cartheight / 4.0
      cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
      self.carttrans = rendering.Transform()
      cart.add_attr(self.carttrans)
      self.viewer.add_geom(cart)
      l,r,t,b = -polewidth / 2,polewidth / 2,polelen - polewidth / 2,-polewidth / 2
      pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
      pole.set_color(.8,.6,.4)
      self.poletrans = rendering.Transform(translation=(0, axleoffset))
      pole.add_attr(self.poletrans)
      pole.add_attr(self.carttrans)
      self.viewer.add_geom(pole)
      self.axle = rendering.make_circle(polewidth / 2)
      self.axle.add_attr(self.poletrans)
      self.axle.add_attr(self.carttrans)
      self.axle.set_color(.5,.5,.8)
      self.viewer.add_geom(self.axle)
      self.track = rendering.Line((0,carty), (screen_width,carty))
      self.track.set_color(0,0,0)
      self.viewer.add_geom(self.track)

    x = self.state
    cartx = x[0] * scale + screen_width / 2.0 # MIDDLE OF CART
    self.carttrans.set_translation(cartx, carty)
    self.poletrans.set_rotation(-x[2])

    return self.viewer.render(return_rgb_array = mode == 'rgb_array')

class CartPoleEnv(ContinuousCartPoleEnv):
 
  def __init__(self):
    
    super(CartPoleEnv, self).__init__()

    # Overwrite the action space with discrete one
    self.action_space = gym.spaces.Discrete(2)

    return 

  def step(self, _action):

    assert self.action_space.contains(_action), "%r (%s) invalid" % (action, type(action))

    # Convert the discrete action to the continuous one
    action = np.array([1.0 if _action == 1 else -1.0])

    return self._base_step(action)



