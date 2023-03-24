import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym


import matplotlib.pyplot as plt
 
BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 2000    


N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0] # 4 states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

        
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(state_count,10)


    def forward(self):
        pass

class dqn(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()


class main():


    def main(self):
        
        # plt.ion()
        # plt.figure(1)
        # t_list = []
        # result_list = []

        # print('main')
                     # The capacity of experience replay buffer

        env = gym.make("CartPole-v1", render_mode="human") # Use cartpole game as environment
        # env = gym.make("CartPole-v1") # Use cartpole game as environment
        env = env.unwrapped
  
        dqn_a = dqn()


        tc = 2**9
        for episode in range(tc):
            state = env.reset()[0]
            print(state)
            ep_r = 0
            while 1:
                action = dqn.choose_action(state)


if __name__ == "__main__":
    main().main()