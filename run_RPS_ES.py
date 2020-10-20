"""
Using NEAT for reinforcement learning.
The detail for NEAT can be find in : http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import neat
import numpy as np
import gym
import visualize

# GAME = 'CartPole-v0'
# env = gym.make(GAME).unwrapped
# from gym_env import CustomEnv
# from simple_env import SimpleEnv
# from xor_env import XorEnv
from maze_env import Maze
# env = CustomEnv()
# env = SimpleEnv()
# env = XorEnv()
env = Maze()

CONFIG = "./config"
EP_STEP = 1000           # maximum episode steps
GENERATION_EP = 10      # evaluate by the minimum of 10-episode rewards
TRAINING = True         # training or testing
CHECKPOINT = 50          # test on this checkpoint
MAX_GENERATION = 100000       # train how many generations

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # net = neat.nn.RecurrentNetwork.create(genome, config)
        ep_r = []
        for ep in range(GENERATION_EP): # run many episodes for the genome in case it's lucky
            accumulative_r = 0.         # stage longer to get a greater episode reward
            # if not TRAINING:
            # print('genome_id', genome_id, 'ep', ep, 'results:', env.results)
            observation = env.reset()
            for t in range(EP_STEP):
                action_values = net.activate(observation)
                # print('action_values', action_values)
                # action = np.argmax(action_values)
                action = np.argmax(action_values)
                observation_, reward, done, _ = env.step(action)
                accumulative_r += reward
                if done:
                    break
                observation = observation_
            ep_r.append(accumulative_r)
        genome.fitness = np.min(ep_r)/float(EP_STEP)    # depends on the minimum episode reward
        # print('genome_id', genome_id, 'fitness', genome.fitness)


def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    # recode history
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    pop.run(eval_genomes, MAX_GENERATION)

    # visualize training
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def evaluation():
    CHECKPOINT = 1759
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    winner = p.run(eval_genomes, 1)     # find the winner in restored population

    # show winner net
    # node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    # visualize.draw_net(p.config, winner, True, node_names=node_names)
    node_names = {-1:'A', -2: 'B', -3: 'C', -4: 'D', 0:'a', 1: 'b', 2: 'c', 3: 'd'}
    visualize.draw_net(p.config, winner, True, node_names=node_names)
    stats = neat.StatisticsReporter()
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    # net = neat.nn.RecurrentNetwork.create(winner, p.config)
    while True:
        s = env.reset()
        done = False
        while not done:
            env.render()
            # a = np.argmax(net.activate(s))
            a = net.activate(s)
            s, r, done, _ = env.step(a)


if __name__ == '__main__':
    if TRAINING:
        run()
    else:
        print('evaluation')
        evaluation()
