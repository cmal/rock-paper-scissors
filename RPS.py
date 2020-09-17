import tensorflow as tf
import numpy as np
import random
import pandas as pd

ACTIONS = ["R", "P", "S"] # rock-paper-scissor R < P, P < S, S < R
EPSILON = 0.1   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

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
    if (np.random.uniform() > EPSILON) or\
       ((state_actions == 0).all()):
      # act non-greedy or state-action have no value
        action = np.random.choice(ACTIONS)
    else:
      # act greedy
      action = state_actions.idxmax()
      # replace argmax to idxmax as argmax means
      # a different function in newer version of pandas
    return action

def get_reward(mine, opponent):
  if mine == opponent:
    return 0
  elif mine == "R" and opponent == "P":
    return -1
  elif mine == "P"and opponent == "S":
    return -1
  elif mine == "S" and opponent == "R":
    return -1
  else:
    return 1
  

def player(prev_play, opponent_history=[]):
  global q, last_action
  if prev_play == "":
    q = build_q(3 * 1, ACTIONS) # use opponents' last 1 choice
    last_action = choose_action(0, q)
    # print("ACTION: ", last_action)
    return last_action
  else:
    opponent_history.append(prev_play)
    state = ACTIONS.index(prev_play)
    last_state = ACTIONS.index(opponent_history[-2]) if len(opponent_history) > 1 else 0
    reward = get_reward(last_action, prev_play)
    q_target = reward + GAMMA * q.iloc[state, :].max()
    q_predict = q.loc[state, last_action]
    q.loc[last_state, last_action] += ALPHA * (q_target - q_predict)

    # print(q)

    # after updated q
    last_action = choose_action(state, q)
    # print("ACTION: ", last_action)
    return last_action