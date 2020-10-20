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



### Learning Algo copied from some book

import  matplotlib
from    matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure()

import  gym,os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    collections import namedtuple
env = RockPaperScissorsEnv()
env.seed(2222)
tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')



gamma = 0.98
epsilon = 0.2
batch_size = 32

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(18, kernel_initializer='he_normal') # I changed 100 to 18
        self.fc2 = layers.Dense(3, kernel_initializer='he_normal') # I changed 4 to 3

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1)
        return x

class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(18, kernel_initializer='he_normal') # I changed 100 to 18
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x




class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = []
        self.actor_optimizer = optimizers.Adam(1e-3)
        self.critic_optimizer = optimizers.Adam(3e-3)

    def select_action(self, s):
        s = tf.constant(s, dtype=tf.float32)
        # s = tf.expand_dims(s, 0)   # I removed this line, otherwise we will get a (1,3,3) tensor and later we will get an error
        prob = self.actor(s)
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a)
        return a, float(prob[0][a])

    def get_value(self, s):
        s = tf.constant(s, dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        v = self.critic(s)[0]
        return float(v)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def optimize(self):
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action,[-1,1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1,1])

        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + gamma * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)

        for _ in range(round(10*len(self.buffer)/batch_size)):

            index = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=False)

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)

                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v
                advantage = tf.stop_gradient(delta)
                a = tf.gather(action, index, axis=0)
                pi = self.actor(tf.gather(state, index, axis=0)) 
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                indices = tf.concat([indices, a], axis=1)
                pi_a = tf.gather_nd(pi, indices)
                pi_a = tf.expand_dims(pi_a, axis=1)
                # Importance Sampling
                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                value_loss = losses.MSE(v_target, v)
            grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        self.buffer = []


def main():
    agent = PPO()
    returns = []
    total = 0
    for i_epoch in range(50000):
        state = env.reset()
        for t in range(games):
            action, action_prob = agent.select_action(state)
            if t == 999:
              print(action, action_prob)
            next_state, reward, done, _ = env.step(action)
            # print(next_state, reward, done, action)
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state
            total += reward
            if done:
                if len(agent.buffer) >= batch_size:
                    agent.optimize()
                break
        print(env.results)

        if i_epoch % 20 == 0:
            returns.append(total/20)
            total = 0
            print(i_epoch, returns[-1])

    print(np.array(returns))
    plt.figure()
    plt.plot(np.arange(len(returns))*20, np.array(returns))
    plt.plot(np.arange(len(returns))*20, np.array(returns), 's')
    plt.xlabel('epochs')
    plt.ylabel('total return')
    plt.savefig('ppo-tf.svg')


if __name__ == '__main__':
    main()
    print("end")
    
