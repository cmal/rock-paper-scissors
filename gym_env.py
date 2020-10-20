import gym
from gym import spaces
from collections import defaultdict
# from stable_baselines.common.env_checker import check_env
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
import numpy as np

ACTIONS = ["R", "P", "S"]
games = 1000
OB_TYPE = 1

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(3)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0.0, high=1000.0,
                                        shape=(3,3), dtype=float)
    # self.observation_space = spaces.Box(low=0.0, high=1000.0,
    #                                     shape=(3,), dtype=float)
    
    self.reset()

  def step(self, actions):
    print('action: ', actions)
    assert actions == 0 or actions == 1 or actions == 2
    opponent_play = self.opponent_play()

    self.prev_plays[self.prev_actions * 3 + actions] += 1
    # print('timestep', self.timestep, 'actions:', actions, 'opponent_play', opponent_play)
    reward = self.calc_reward(actions, opponent_play)
    terminal = False

    self.calc_state(self.timestep, opponent_play, actions)
    ob = self.get_ob()
    print('ob: ', ob)
    self.prev_actions = actions
    self.prev_opponent_play = opponent_play
    self.timestep += 1
    # return self.state, terminal, reward  # 第一个参数返回states用于下一步agent.act传入使用
    return ob, reward, terminal, None

  def reset(self):
    self.opponent = abbey
    self.timestep = 0
    self.prev_opponent_play = 0
    self.prev_actions = 0
    self.prev_plays = defaultdict(int)
    # self.init_state = np.ones(3, dtype=float) * 1.0 / 3.0
    self.init_state = np.zeros((3,3), dtype=int)
    self.state = np.copy(self.init_state)
    self.results = {"win": 0, "lose": 0, "tie": 0}
    return self.get_ob()  # reward, done, info can't be included

  def render(self, mode='human'):
    pass

  def close (self):
    pass

  def calc_reward(self, actions, play):
    if self.timestep % games == games - 1:
      # print(self.prev_plays)
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
    # print('opp play: ', opp_play)
    return ACTIONS.index(opp_play)

  def calc_state(self, timestep, opponent_play, actions):
    # # print('prev_plays', self.prev_plays)
    # a = self.prev_plays[self.prev_actions * 3]
    # b = self.prev_plays[self.prev_actions * 3 + 1]
    # c = self.prev_plays[self.prev_actions * 3 + 2]
    # s = a + b + c
    # self.state = np.array([float(a)/s, float(b)/s, float(c)/s]) if s else np.copy(self.init_state)
    self.state[self.prev_actions][actions] += 1

  def get_ob(self):
    # print(self.state)
    if OB_TYPE == 0:
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
      return ob.flatten()
    elif OB_TYPE == 1:
      return self.state.flatten()
    elif OB_TYPE == 2:
      return np.array([self.prev_actions])


import  matplotlib
from 	matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
# matplotlib.rcParams['font.family'] = ['Monaco']
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure()

import  gym,os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    collections import namedtuple
# from    torch.utils.data import SubsetRandomSampler,BatchSampler
# from .gym_env import CustomEnv
# env = gym.make('CartPole-v1')  # 创建游戏环境
env = CustomEnv()
# check_env(env)
env.seed(2222)
tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')



gamma = 0.98 # 激励衰减因子
epsilon = 0.2 # PPO误差超参数0.8~1.2
batch_size = 32 # batch size


# 创建环境
# env = gym.make('CartPole-v0').unwrapped
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        # 策略网络，也叫Actor网络，输出为概率分布pi(a|s)
        self.fc1 = layers.Dense(18, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(3, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1) # 转换成概率
        return x

class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        # 偏置b的估值网络，也叫Critic网络，输出为v(s)
        self.fc1 = layers.Dense(18, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x




class PPO():
    # PPO算法主体
    def __init__(self):
        super(PPO, self).__init__()
        self.actor = Actor() # 创建Actor网络
        self.critic = Critic() # 创建Critic网络
        self.buffer = [] # 数据缓冲池
        self.actor_optimizer = optimizers.Adam(1e-3) # Actor优化器
        self.critic_optimizer = optimizers.Adam(3e-3) # Critic优化器

    def select_action(self, s):
        # 送入状态向量，获取策略: [4]
        s = tf.constant(s, dtype=tf.float32)
        # print(s.shape)
        # s: [4] => [1,4]
        # s = tf.expand_dims(s, 0)
        # 获取策略分布: [1, 2]
        prob = self.actor(s)
        # 从类别分布中采样1个动作, shape: [1]
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a)  # Tensor转数字
        return a, float(prob[0][a]) # 返回动作及其概率

    def get_value(self, s):
        # 送入状态向量，获取策略: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1,4]
        s = tf.expand_dims(s, axis=0)
        # 获取策略分布: [1, 2]
        v = self.critic(s)[0]
        return float(v) # 返回v(s)

    def store_transition(self, transition):
        # 存储采样数据
        self.buffer.append(transition)

    def optimize(self):
        # 优化网络主函数
        # 从缓存中取出样本数据，转换成Tensor
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action,[-1,1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1,1])
        # 通过MC方法循环计算R(st)
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + gamma * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)
        # 对缓冲池数据大致迭代10遍
        for _ in range(round(10*len(self.buffer)/batch_size)):
            # 随机从缓冲池采样batch size大小样本
            index = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=False)
            # 构建梯度跟踪环境
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # 取出R(st)，[b,1]
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                # 计算v(s)预测值，也就是偏置b，我们后面会介绍为什么写成v
                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v # 计算优势值
                advantage = tf.stop_gradient(delta) # 断开梯度连接 
                # 由于TF的gather_nd与pytorch的gather功能不一样，需要构造
                # gather_nd需要的坐标参数，indices:[b, 2]
                # pi_a = pi.gather(1, a) # pytorch只需要一行即可实现
                a = tf.gather(action, index, axis=0) # 取出batch的动作at
                # batch的动作分布pi(a|st)
                pi = self.actor(tf.gather(state, index, axis=0)) 
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                indices = tf.concat([indices, a], axis=1)
                pi_a = tf.gather_nd(pi, indices)  # 动作的概率值pi(at|st), [b]
                pi_a = tf.expand_dims(pi_a, axis=1)  # [b]=> [b,1] 
                # 重要性采样
                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
                # PPO误差函数
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # 对于偏置v来说，希望与MC估计的R(st)越接近越好
                value_loss = losses.MSE(v_target, v)
            # 优化策略网络
            grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            # 优化偏置值网络
            grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        self.buffer = []  # 清空已训练数据


def main():
    agent = PPO()
    returns = [] # 统计总回报
    total = 0 # 一段时间内平均回报
    for i_epoch in range(500): # 训练回合数
        state = env.reset() # 复位环境
        for t in range(games): # 最多考虑 games 步
            # 通过最新策略与环境交互
            action, action_prob = agent.select_action(state)
            if t == 999:
              print(action, action_prob)
            next_state, reward, done, _ = env.step(action)
            # print(next_state, reward, done, action)
            # 构建样本并存储
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state # 刷新状态
            total += reward # 累积激励
            if done: # 合适的时间点训练网络
                if len(agent.buffer) >= batch_size:
                    agent.optimize() # 训练网络
                break
        print(env.results)

        if i_epoch % 20 == 0: # 每20个回合统计一次平均回报
            returns.append(total/20)
            total = 0
            print(i_epoch, returns[-1])

    print(np.array(returns))
    plt.figure()
    plt.plot(np.arange(len(returns))*20, np.array(returns))
    plt.plot(np.arange(len(returns))*20, np.array(returns), 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('ppo-tf-cartpole.svg')


if __name__ == '__main__':
    main()
    print("end")
    
