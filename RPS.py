import tensorflow as tf
import numpy as np
import random
import pandas as pd
import time
from itertools import product

ACTIONS = ["R", "P", "S"] # rock-paper-scissor R < P, P < S, S < R
EPSILON = 0.97   # greedy police
ALPHA = 0.05     # learning rate
# GAMMA = 0.9    # discount factor
GAMMA = 0.9    # discount factor
# MAX_EPISODES = 13   # maximum episodes
# FRESH_TIME = 0.3    # fresh time for one move

def build_q(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table

def choose_action(state, q):
    # This is how to choose an action
    state_actions = q.iloc[state, :]
    # print(np.random.uniform() > EPSILON)
    # print(state_actions)
    # print((state_actions == 0).all())
    if (np.random.uniform() > EPSILON) or\
       ((state_actions == 0).all()):
      # act non-greedy or state-action have no value
      # print('use random')
      action = np.random.choice(ACTIONS)
    else:
      # greedy
      action = state_actions.idxmax()

      # # random
      # max_val = state_actions.argmax()
      # next_action = state_actions[state_actions != max_val].idxmax()
      # next_max = state_actions[state_actions != max_val].argmax()
      # is_near = max_val <= next_max * 1.2 and max_val > 0 and next_max > 0

      # if (np.random.uniform() > 0.5 and is_near):
      #   # print("use random largest 2", action, state_actions[state_actions != state_actions.argmax()])
      #   action = next_action

    return action

def get_reward(mine, opponent):
  if mine == opponent:
    return 0.3
  elif mine == "R" and opponent == "P":
    return -0.1
  elif mine == "P"and opponent == "S":
    return -0.1
  elif mine == "S" and opponent == "R":
    return -0.1
  else:
    return 0.3

OLD = 3
STATE_SPACE = [''.join(pair) for pair in product(["R", "P", "S"], repeat=OLD)]

def get_state_index(last_two):
  return STATE_SPACE.index(last_two)
  
def player3(prev_play, opponent_history=[]):
  global q, last_action
  if prev_play == "":
    q = build_q(pow(3, OLD) * 1, ACTIONS) # use opponents' last OLD choices
    last_action = choose_action(0, q)
    # print("ACTION: ", last_action)
    return last_action
  else:
    opponent_history.append(prev_play)
    last_three = ''.join(opponent_history[-(OLD + 1):]) if len(opponent_history) > (OLD + 1) else 'R' * (OLD + 1)
    state = get_state_index(last_three[1:])
    last_state = get_state_index(last_three[:-1])
    reward = get_reward(last_action, prev_play)
    # print(q)
    # print("LAST_STATE: ", last_state)
    # print("STATE: ", state)
    # print("LAST_ACTION: ", last_action)
    # print("REWARD: ", reward)
    # time.sleep(1)


    # off-policy, q-learning
    #q_target = reward + GAMMA * q.iloc[state, :].max()

    # on-policy, sarsa
    action = choose_action(state, q)
    q_target = reward + GAMMA * q.iloc[state, q.columns.get_loc(action)]


    q_predict = q.loc[state, last_action]
    # print((q_target - q_predict))
    q.loc[last_state, last_action] += ALPHA * (q_target - q_predict)

    # if len(opponent_history) % 500 == 0:
    #   print(q)

    # off policy
    # last_action = choose_action(state, q)

    # on policy
    last_action = action
    

    # print("ACTION: ", last_action)
    return last_action

def player0(prev_play, opponent_history=[]):
  opponent_history.append(prev_play)
  guess = "R"
  if len(opponent_history) > 2:
    guess = opponent_history[-2]
  return guess # if guess != "" else "R"


def player1(prev_play, opponent_history=[]):
  #opponent_history.append(prev_play)
  # guess = "R"
  # if len(opponent_history) > 2:
  #   guess = opponent_history[-2]
  return "R"


def player2(prev_play, opponent_history=[]):
  opponent_history.append(prev_play)
  # guess = "R"
  # if len(opponent_history) > 2:
  #   guess = opponent_history[-2]
  return random.choice(ACTIONS)


def player(*args, **kwargs):
  # return player0(*args, **kwargs)
  # return player1(*args, **kwargs)
  # return player2(*args, **kwargs)
  return player3(*args, **kwargs)
  # return player4(*args, **kwargs)
