import gym
from gym import spaces
from collections import defaultdict
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
import numpy as np
from random import choice, randrange

class XorEnv(gym.Env):
  """Simple Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(XorEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low=1.0, high=2000.0, shape=(1,), dtype=np.float32)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=1.0, high=1000.0, shape=(2,), dtype=np.float32)
    self.reset()

  def step(self, actions):
    # print('action: ', actions)
    assert actions == 0 or actions == 1
    # opponent_play = self.opponent_play()
    # print('actions', actions)
    # print('opponent_play', opponent_play)
    reward = self.calc_reward(actions)
    # print('reward', reward)
    terminal = False

    ob = self.get_ob()
    # print('ob: ', ob, 'prev_actions', actions)
    self.prev_actions = actions
    self.timestep += 1
    return ob, reward, terminal, None

  def reset(self):
    self.timestep = 0
    self.prev_actions = 0
    self.results = {"win": 0, "lose": 0}
    return self.get_ob()  # reward, done, info can't be included

  def render(self, mode='human'):
    pass

  def close (self):
    pass

  def calc_reward(self, actions):
    # print('calc_reward, actions: ', actions, 'play: ', play)
    # if actions != self.a * self.b:
    #   self.results['lose'] += 1
    #   # print('reward -1')
    #   return -1.0
    # else:
    #   self.results['win'] += 1
    #   # print('reward +1')
    #   return 1.0
    return -abs(self.a + self.b - actions)

  # def opponent_play(self):
  #   # return self.timestep % 2
  #   # return 1 - self.prev_actions
  #   return self.timestep > 300 and self.timestep < 350

  def get_ob(self):
    self.a = randrange(1.0,1000.0)
    self.b = randrange(1.0,1000.0)
    return np.array([self.a, self.b])
    # return np.array([
    #   self.prev_actions,
    #   self.timestep
    # ])
