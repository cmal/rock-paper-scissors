import random
import utils
import TD3
import OurDDPG
import DDPG
import numpy as np

state_dim = 4

kwargs = {
    'state_dim': state_dim,
    'action_dim': 1,
    'max_action': 3.0,
    'discount': 0.9,
    'tau': 0.2,
    'policy_noise': 0.3,
    'noise_clip': 1.0,
    'policy_freq': 2
}

# OLD = 3
# STATE_SPACE = [''.join(pair) for pair in product(["R", "P", "S"], repeat=OLD)]
ACTIONS = ["R", "P", "S"] # rock-paper-scissor R < P, P < S, S < R

# def get_state_index(last_two):
#   return STATE_SPACE.index(last_two)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

policy = "DDPG"
    
def init_player():
  global policy
  if policy == "TD3":
      # Target policy smoothing is scaled wrt the action scale
      kwargs["policy_noise"] = kwargs['policy_noise'] * kwargs['max_action']
      kwargs["noise_clip"] = kwargs['noise_clip'] * kwargs['max_action']
      kwargs["policy_freq"] = kwargs['policy_freq']
      policy = TD3.TD3(**kwargs)
  elif policy == "OurDDPG":
      del kwargs['policy_noise']
      del kwargs['noise_clip']
      del kwargs['policy_freq']
      policy = OurDDPG.DDPG(**kwargs)
  elif policy == "DDPG":
      del kwargs['policy_noise']
      del kwargs['noise_clip']
      del kwargs['policy_freq']
      policy = DDPG.DDPG(**kwargs)

def get_reward(mine, opponent):
  mine = ACTIONS[int(mine[0])]
  if mine == opponent:
    return 0.1
  elif mine == "R" and opponent == "P":
    return -0.3
  elif mine == "P"and opponent == "S":
    return -0.3
  elif mine == "S" and opponent == "R":
    return -0.3
  else:
    return 0.3

def get_state(choices):
    choices = choices[-state_dim:]
    return np.array(
        [ACTIONS.index(x) for x in choices if x != ""]
    )

def player(prev_play, opponent_history=[]):
  opponent_history.append(prev_play)
  global replay_buffer, state, prev_action
  if prev_play == "":
    init_player()
    replay_buffer = utils.ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'])
    # opponent_history = []
    state = []

  if len(opponent_history) % 1000 < state_dim+2:
    choice = random.choice(ACTIONS)
    prev_action = [ACTIONS.index(choice)]
    state = get_state(opponent_history)
    return choice

  prev_state = state
  state = get_state(opponent_history)
  
  prev_reward = get_reward(prev_action, prev_play)
  #print(prev_state, prev_action, state, prev_reward, True)
  replay_buffer.add(prev_state, prev_action, state, prev_reward, True)

  policy.train(replay_buffer, 10)
  prev_action = policy.select_action(np.array(state)) # I didn't add noise
  # print("action", prev_action)
  return ACTIONS[int(abs(prev_action[0]))] # use continous for discrete
  # episode_reward += reward

  # if len(opponent_history) % 10 == 0:
    

  # if done:
  #     state = []
  #     done = False
  #     episode_reward = 0
  
