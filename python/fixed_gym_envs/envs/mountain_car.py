"""
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""

import numpy as np

import gym
import gym.spaces
from gym.utils import seeding

class ContinuousMountainCarEnv(gym.Env):

  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self):
    self.min_position = -1.2
    self.max_position = 0.6
    self.max_speed = 0.07
    self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    self.power = 0.0015

    self.min_acceleration = -1.0
    self.max_acceleration = 1.0

    self.low = np.array([self.min_position, -self.max_speed])
    self.high = np.array([self.max_position, self.max_speed])

    self.accel_low = np.array([self.min_acceleration])
    self.accel_high = np.array([self.max_acceleration])

    self.viewer = None

    self.action_space = gym.spaces.Box(self.accel_low, self.accel_high)
    self.observation_space = gym.spaces.Box(self.low, self.high)

    self.seed()
    self.reset()

    self.step_counter = 0

    # Due to inconsitencies in the TimeWrapper we cannot rely on this one
    # Thus we introduce a custom limit in the world which terminates an episode
    self.max_episode_steps = 2500

  def _height(self, xs):
    return np.sin(3 * xs) * .45 + .55

  def _base_step(self, _action, free_fuel=True):
    
    self.step_counter += 1

    position = self.state[0]
    velocity = self.state[1]
    
    force = _action[0] # Get scalar to remove shape issues

    velocity += force * self.power - 0.0025 * np.cos(3 * position)
    if (velocity > self.max_speed): velocity = self.max_speed
    if (velocity < -self.max_speed): velocity = -self.max_speed

    position += velocity
    if (position > self.max_position): position = self.max_position
    if (position < self.min_position): position = self.min_position
    if (position == self.min_position and velocity < 0): velocity = 0

    goal_reached = position >= self.goal_position
    time_is_up = self.step_counter >= self.max_episode_steps
    
    done = goal_reached or time_is_up

    if done:
      reward = 0.0 if goal_reached else -1.0 
    else:
      reward = -1.0
    
    if not free_fuel:
      reward -= 0.1 * force ** 2
    
    self.state = np.array([position, velocity])

    return self.state, reward, done, {}

  # Implementation of interface

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, _action):
    
    assert self.action_space.contains(_action), "%r (%s) invalid" % (_action, type(_action))
    
    return self._base_step(_action)

  def reset(self):
    self.step_counter = 0
    self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
    return np.array(self.state)

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    screen_width = 600
    screen_height = 400

    world_width = self.max_position - self.min_position
    scale = screen_width / world_width
    carwidth = 40
    carheight = 20


    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      xs = np.linspace(self.min_position, self.max_position, 100)
      ys = self._height(xs)
      xys = zip((xs - self.min_position) * scale, ys * scale)

      self.track = rendering.make_polyline(xys)
      self.track.set_linewidth(4)
      self.viewer.add_geom(self.track)

      clearance = 10

      l,r,t,b = -carwidth / 2, carwidth / 2, carheight, 0
      car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
      car.add_attr(rendering.Transform(translation=(0, clearance)))
      self.cartrans = rendering.Transform()
      car.add_attr(self.cartrans)
      self.viewer.add_geom(car)
      frontwheel = rendering.make_circle(carheight / 2.5)
      frontwheel.set_color(.5, .5, .5)
      frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4,clearance)))
      frontwheel.add_attr(self.cartrans)
      self.viewer.add_geom(frontwheel)
      backwheel = rendering.make_circle(carheight / 2.5)
      backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4,clearance)))
      backwheel.add_attr(self.cartrans)
      backwheel.set_color(.5, .5, .5)
      self.viewer.add_geom(backwheel)
      flagx = (self.goal_position - self.min_position) * scale
      flagy1 = self._height(self.goal_position) * scale
      flagy2 = flagy1 + 50
      flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
      self.viewer.add_geom(flagpole)
      flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
      flag.set_color(.8,.8,0)
      self.viewer.add_geom(flag)

    pos = self.state[0]
    self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
    self.cartrans.set_rotation(np.cos(3 * pos))

    return self.viewer.render(return_rgb_array = mode == 'rgb_array')

class MountainCarEnv(ContinuousMountainCarEnv):

  def __init__(self):
    super(MountainCarEnv, self).__init__()

    # Overwrite the action space with discrete one
    self.action_space = gym.spaces.Discrete(3)

    return 

  def step(self, _action):
        
    assert self.action_space.contains(_action), "%r (%s) invalid" % (action, type(action))

    # Move a = 0,1,2 into correct range
    action = np.array([_action - 1])

    return self._base_step(action)
