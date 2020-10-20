### define a Rock-Paper-Scissor opponent

abbey_state = []
play_order=[{
              "RR": 0,
              "RP": 0,
              "RS": 0,
              "PR": 0,
              "PP": 0,
              "PS": 0,
              "SR": 0,
              "SP": 0,
              "SS": 0,
          }]
def abbey(prev_opponent_play,
          re_init=False):
    if not prev_opponent_play:
        prev_opponent_play = 'R'
    global abbey_state, play_order
    if re_init:
        abbey_state = []
        play_order=[{
              "RR": 0,
              "RP": 0,
              "RS": 0,
              "PR": 0,
              "PP": 0,
              "PS": 0,
              "SR": 0,
              "SP": 0,
              "SS": 0,
          }]
    abbey_state.append(prev_opponent_play)
    last_two = "".join(abbey_state[-2:])
    if len(last_two) == 2:
        play_order[0][last_two] += 1
    potential_plays = [
        prev_opponent_play + "R",
        prev_opponent_play + "P",
        prev_opponent_play + "S",
    ]
    sub_order = {
        k: play_order[0][k]
        for k in potential_plays if k in play_order[0]
    }
    prediction = max(sub_order, key=sub_order.get)[-1:]
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prediction]


### define the gym env
import gym
from gym import spaces
from collections import defaultdict
import numpy as np

ACTIONS = ["R", "P", "S"]
games = 1000

class RockPaperScissorsEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(RockPaperScissorsEnv, self).__init__()
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(low=0.0, high=1.0,
                                        shape=(3,3), dtype=float)
    self.reset()

  def step(self, actions):
    assert actions == 0 or actions == 1 or actions == 2
    opponent_play = self.opponent_play()

    self.prev_plays[self.prev_actions * 3 + actions] += 1
    reward = self.calc_reward(actions, opponent_play)
    terminal = False

    self.calc_state(self.timestep, opponent_play, actions)
    self.prev_actions = actions
    self.prev_opponent_play = opponent_play
    self.timestep += 1
    return self.get_ob(), reward, terminal, None

  def reset(self):
    self.opponent = abbey
    self.timestep = 0
    self.prev_opponent_play = 0
    self.prev_actions = 0
    self.prev_plays = defaultdict(int)
    self.init_state = np.zeros((3,3), dtype=int)
    # the internal state
    self.state = np.copy(self.init_state)
    self.results = {"win": 0, "lose": 0, "tie": 0}
    return self.get_ob()

  def render(self, mode='human'):
    pass

  def close (self):
    pass

  def calc_reward(self, actions, play):
    if self.timestep % games == games - 1:
      pass
    if actions == play:
      self.results['tie'] += 1
      return 0
    elif actions == 0 and play == 1:
      self.results['lose'] += 1
      return -0.3
    elif actions == 1 and play == 2:
      self.results['lose'] += 1
      return -0.3
    elif actions == 2 and play == 0:
      self.results['lose'] += 1
      return -0.3
    elif (actions == 1 and play == 0) or (actions == 2 and play == 1) or (actions == 0 and play == 2):
      self.results['win'] += 1
      return 0.3
    else:
      raise NotImplementedError('calc_reward something get wrong')

  def opponent_play(self):
    re_init = (self.timestep == 0)
    opp_play = self.opponent(ACTIONS[self.prev_actions], re_init=re_init)
    return ACTIONS.index(opp_play)

  def calc_state(self, timestep, opponent_play, actions):
    self.state[self.prev_actions][actions] += 1

  def get_ob(self):
    '''return observations'''
    state0 = self.state[0]
    sum0 = state0.sum()
    state1 = self.state[1]
    sum1 = state1.sum()
    state2 = self.state[2]
    sum2 = state2.sum()
    init = np.ones(3, dtype=float) / 3.0
    ob = np.array([
      state0 / sum0 if sum0 else init,
      state1 / sum1 if sum1 else init,
      state2 / sum2 if sum2 else init,
    ])
    # print(ob)
    return ob
