from tensorforce.environments import Environment
from tensorforce.agents import Agent
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
import numpy as np
from collections import defaultdict

ACTIONS = ["R", "P", "S"]
games = 1000
dim = 2

class RockPaperScissorsEnvironment(Environment):
    def __init__(self):
        self.timestep = 0
        self.prev_opponent_play = 0
        self.prev_actions = 0
        self.prev_plays = defaultdict(int)
        self.opponent = None
        # self.init_state = -np.zeros((dim,), dtype=int)
        self.init_state = np.ones(3, dtype=float) * 1.0 / 3.0
        self.state = np.copy(self.init_state)
        super().__init__()


    def states(self):  # observations
        return dict(type='float', shape=(3,), min_value=0.0, max_value=1.0)

    def actions(self):
        return dict(type='int', num_values=3)


    # # Optional, should only be defined if environment has a natural maximum
    # # episode length
    # def max_episode_timesteps(self):
    #     return super().max_episode_timesteps()


    # # Optional
    def close(self):
        super().close()


    def reset(self):
        """Reset state.
        """
        # self.timestep = 0
        # self.current_temp = np.random.random(size=(1,))
        # return self.current_temp
        # print('reset', self.init_state)
        self.timestep = 0
        self.prev_plays = defaultdict(int)
        self.state = np.copy(self.init_state) # reset() 的返回值应该赋值给 states
        return self.state


    def calc_reward(self, actions, play):
        if self.timestep % games == games - 1:
            # print(self.prev_plays)
            pass
        if actions == play:
            return 0
        elif actions == 0 and play == 1:
            return -0.3
        elif actions == 1 and play == 2:
            return -0.3
        elif actions == 2 and play == 0:
            return -0.3
        elif (actions == 1 and play == 0) or (actions == 2 and play == 1) or (actions == 0 and play == 2):
            return 0.3
        else:
            print('calc_reward something get wrong', actions, play)

    def opponent_play(self):
        re_init = (self.timestep == 0)
        return ACTIONS.index(self.opponent(ACTIONS[self.prev_actions], re_init=re_init))

    def calc_state(self, timestep, opponent_play, actions):
        a = self.prev_plays[self.prev_actions * 3]
        b = self.prev_plays[self.prev_actions * 3 + 1]
        c = self.prev_plays[self.prev_actions * 3 + 2]
        s = a + b + c
        # print(a,b,c,s)
        self.state = np.array([float(a)/s, float(b)/s, float(c)/s]) if s else np.copy(self.init_state)

    def execute(self, actions):
        ## Check the action is either 0 or 1 -- heater on or off.
        assert actions == 0 or actions == 1 or actions == 2
        # if self.step % games == 0:
        #   self.change_opponent()
        ## play
        opponent_play = self.opponent_play()

        ## Compute the reward
        self.prev_plays[self.prev_actions * 3 + actions] += 1
        reward = self.calc_reward(actions, opponent_play)
        # print(reward)

        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == False means episode is not done
        ## terminal == True means it is done.
        terminal = False

        # self.state[self.step % games] = opponent_play
        self.calc_state(self.timestep, opponent_play, actions)
        self.prev_actions = actions
        self.prev_opponent_play = opponent_play
        self.timestep += 1

        return self.state, terminal, reward  # 第一个参数返回states用于下一步agent.act传入使用

###-----------------------------------------------------------------------------
### Create the environment
###   - Tell it the environment class
###   - Set the max timestamps that can happen per episode
environment = Environment.create(
    environment=RockPaperScissorsEnvironment,
    max_episode_timesteps=1000,
  )

def evaluate(environment, agent, states):
    terminal = False
    internals = agent.initial_internals()
    while not terminal and environment.timestep < games:
        # print(states.shape)
        actions, internals = agent.act(states=states, internals=internals, independent=True)
        # print('action', ACTIONS[actions])
        states, terminal, reward = environment.execute(actions=actions)
        # print(states)

    return agent

if __name__ == "__main__":
    states = environment.reset()
    # environment.opponent = [quincy, abbey, kris, mrugesh][self.step // games]
    # environment.opponent = quincy
    environment.opponent = abbey
    # environment.opponent = kris
    # environment.opponent = mrugesh
    agent = Agent.create(
      agent='tensorforce',
      # agent='ppo.json',
      # agent='a2c',
      environment=environment,
      update=1,
      # optimizer=dict(optimizer='adam', learning_rate=1e-3),
      optimizer=dict(optimizer='adam', learning_rate=0.1),
      objective='policy_gradient',
      policy=dict(
        type="parametrized_distributions",
        # network=[
        #   dict(type='dense', size=dim, activation='tanh'),
        #   dict(type='dense', size=8, activation='tanh'),
        # ],
        # distributions=dict(
        #   float=dict(type='categorical'),
        # ),
        # temperature=dict(
        #   type='decaying',
        #   decay='exponential',
        #   unit='episodes',
        #   num_steps=100,
        #   initial_value=0.01,
        #   decay_rate=0.5
        # )
      ),
      reward_estimation=dict(horizon=dict(
        type='linear', unit='episodes', num_steps=1000,
        initial_value=10, final_value=50
      )),
      # save agent
      saver=dict(
        directory='data/checkpoints/' + environment.opponent.__name__,
        frequency=20  # save checkpoint every 100 updates
      ),
      # tensorboard view
      summarizer=dict(
        directory='data/summaries/' + environment.opponent.__name__,
      ),
    )
    evaluate(environment, agent, states)
    for _ in range(200):
        sum_reward = 0.0
        states = environment.reset()
        terminal = False
        while not terminal and environment.timestep < games:
            # print('observe state', states)
            actions = agent.act(states=states)
            # print('chose actions:', actions)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            sum_reward += reward
        print('sum_reward', sum_reward)

    states = environment.reset()
    evaluate(environment, agent, states)


