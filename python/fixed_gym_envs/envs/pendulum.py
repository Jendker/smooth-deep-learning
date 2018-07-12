
import os
import numpy as np

import gym
import gym.spaces
from gym.utils import seeding

class PendulumEnv(gym.Env):
  metadata = {
    'render.modes' : ['human', 'rgb_array'],
    'video.frames_per_second' : 30
  }

  def __init__(self):
    self.max_speed = 8
    self.max_torque = 2.
    self.dt = .05
    self.viewer = None

    high = np.array([1., 1., self.max_speed])
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    self.observation_space = gym.spaces.Box(low=-high, high=high)

    self.seed()

    self.step_counter = 0

    # Due to inconsitencies in the TimeWrapper we cannot rely on this one
    # Thus we introduce a custom limit in the world which terminates an episode
    self.max_episode_steps = 500

    return

  def _angle_normalize(self,x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

  def _get_obs(self, state):
    theta, thetadot = state
    return np.array([np.cos(theta), np.sin(theta), thetadot])

  # Implementation of interface
  
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self,_action):

    assert self.action_space.contains(_action), "%r (%s) invalid" % (_action, type(_action))

    self.step_counter += 1

    th, thdot = self.state # th := theta

    u = _action[0] # Get scalar to remove shape issues

    g = 10.
    m = 1.
    l = 1.
    dt = self.dt

    u = u * self.max_torque # Original action space was from -max torque to max_torque, now its -1 to 1 hence scale it
    self.last_u = u # for rendering
    
    #u = np.clip(u, -self.max_torque, self.max_torque)[0]

    costs = self._angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

    time_is_up = self.step_counter >= self.max_episode_steps
    done = time_is_up

    self.state = np.array([newth, newthdot])
    return self._get_obs(self.state), -costs, done, {}

  def reset(self):
    self.step_counter = 0

    high = np.array([np.pi, 1])
    self.state = self.np_random.uniform(low=-high, high=high)
    self.last_u = None
    return self._get_obs(self.state)

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(500,500)
      self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
      rod = rendering.make_capsule(1, .2)
      rod.set_color(.8, .3, .3)
      self.pole_transform = rendering.Transform()
      rod.add_attr(self.pole_transform)
      self.viewer.add_geom(rod)
      axle = rendering.make_circle(.05)
      axle.set_color(0,0,0)
      self.viewer.add_geom(axle)
      fname = os.path.join(os.path.dirname(__file__), "assets/clockwise.png")
      self.img = rendering.Image(fname, 1., 1.)
      self.imgtrans = rendering.Transform()
      self.img.add_attr(self.imgtrans)

    self.viewer.add_onetime(self.img)
    self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
    if self.last_u:
      self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

    return self.viewer.render(return_rgb_array = mode == 'rgb_array')

