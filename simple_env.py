import gym
from gym import spaces
from collections import defaultdict
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
import numpy as np

class SimpleEnv(gym.Env):
  """Simple Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(SimpleEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(2)
    # Example for using image as input:
    self.observation_space = spaces.Discrete(1000)
    self.reset()

  def step(self, actions):
    # print('action: ', actions)
    assert actions == 0 or actions == 1
    opponent_play = self.opponent_play()
    # print('actions', actions)
    # print('opponent_play', opponent_play)
    reward = self.calc_reward(actions, opponent_play)
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

  def calc_reward(self, actions, play):
    # print('calc_reward, actions: ', actions, 'play: ', play)
    if actions == play:
      self.results['lose'] += 1
      # print('reward -1')
      return -1.0
    else:
      self.results['win'] += 1
      # print('reward +1')
      return 1.0

  def opponent_play(self):
    # return self.timestep % 2
    # return 1 - self.prev_actions
    return self.timestep > 300 and self.timestep < 350

  def get_ob(self):
    return np.array([self.timestep])
    # return np.array([
    #   self.prev_actions,
    #   self.timestep
    # ])
