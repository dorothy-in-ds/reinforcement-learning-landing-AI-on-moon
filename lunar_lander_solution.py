import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch
import random

from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make('LunarLander-v2')
env.seed(1998)
np.random.seed(1998)
env.action_space.seed(1998)
torch.manual_seed(1998)

#network initiated similar to 3/10/2021 OH
#todo: try two layers
class network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # hidden layer
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        #output layer
        self.fc3 = nn.Linear(32,output_size)
    def forward(self, x):
        # Get intermediate outputs using hidden layer
        #print('input to model', x)
        x = self.relu1(self.fc1(x))
        #print('forward one layer', x)
        x = self.relu2(self.fc2(x))
        #print('forward two layers', x)
        # Get predictions using output layer
        x = self.fc3(x)
        #print('output layer', x)
        return x


def learn_experience(d,q_net,q_net_target,minibatch_size,gamma,lr):
    #network settings
    optimizer = torch.optim.SGD(q_net.parameters(), lr=lr)
    minibatch = random.sample(d,minibatch_size) 
    next_states = np.array([x.s_next for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    done = np.array([x.done for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    states = np.array([x.s for x in minibatch])
    #calculate y for minibatch transitions
    next_states_values = q_net_target(torch.Tensor(next_states)).max(1).values.detach().numpy()
#     print('rewards: ',rewards)
#     print('done: ',done)
#     print('next_states_values: ',next_states_values)
    q_values_y = torch.Tensor(rewards + (gamma*(1-done)*next_states_values))
    #get real q value
    states = torch.Tensor(states)
    states_values = q_net(states)
#     print('states_values: ',states_values)
    actions = torch.from_numpy(actions[:,None])
#     print('actions: ',actions)
    q_values = states_values.gather(1,actions).squeeze()
#     print('q_values: ',q_values)
#     print('q_values_y: ',q_values_y)
    #calculate the loss
    optimizer.zero_grad()
    loss = F.mse_loss(q_values,q_values_y)
    #perform gradient descent step on with respect to the network parameters (action-values in q)
    loss.backward()
    optimizer.step()
    return q_net


def train_dqn(q_net,q_net_target,n_episodes = 1000,gamma = 0.98,epsilon = 0.7,lr = 0.001,converged = False,minibatch_size = 64):
    #hyperparameters
    #Initialize replay memory to capacity N
    c_steps = 5 ### update target q net every c steps
    transition = namedtuple('transition',['s', 'action', 'reward', 's_next', 'done'])
    experiences = []
    scores = []
    n_steps = []
    episode = 0
    for _ in range(n_episodes):
        episode += 1
        if(episode%1000 == 0): print('episode: ', episode)
        print(episode)
        #initialize the first state and preprocess it 
        s = env.reset()
        #print('state is: ',s)
        preprocessed_s = torch.from_numpy(s).unsqueeze(0)
        #print('preprocessed state is: ',preprocessed_s)
        done = False
        step = 0
        total_reward = 0
        while not done:
            step += 1
            #epsilon greedy method to choose the action
            if  np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                predictions = q_net(preprocessed_s).detach().numpy()
                #print('state_action_values are: ',predictions)
                a = np.argmax(predictions)
            #print('the action chosen is:', a)
            s_next,reward,done,info = env.step(a)
            total_reward += reward
            preprocessed_s_next = torch.from_numpy(s).unsqueeze(0)
            #store transition in d. 
            #if false done due to max steps reached then change done to false
            if done and step == env._max_episode_steps:
                t = transition(s,a,reward,s_next,False)
            else: 
                t = transition(s,a,reward,s_next,done)
            experiences = experiences + [t]
            #sample and learn a minibatch from experiences
            if (len(experiences)>=minibatch_size):
                q_net.load_state_dict(learn_experience(experiences,q_net,q_net_target,minibatch_size,gamma,lr).state_dict())
            if step % c_steps == 0:
                q_net_target.load_state_dict(q_net.state_dict())
            s = s_next
            preprocessed_s = preprocessed_s_next
        scores = scores + [total_reward]
        n_steps = n_steps +[step]
        #if the past 100 scores have an average of 200 points then break
#         if sum(scores[-100:])/100 >= 200: 
#             Converged = True
#             break
        #decay epsilon so that more exploration is allowed at the beginning and more exploitation is used later on when number pf episodes get big
        epsilon *= 0.995
        epsilon = max(epsilon, 0.05) 
    return scores, n_steps, episode, converged

#network settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#played the trained agent
def play_dqn(q_net,n_episodes=500):
    scores = []
    n_steps = []
    episode = 0
    for _ in range(n_episodes):
        episode += 1
        print(episode)
        #initialize the first state and preprocess it 
        s = env.reset()
        #print('state is: ',s)
        preprocessed_s = torch.from_numpy(s).unsqueeze(0)
        #print('preprocessed state is: ',preprocessed_s)
        done = False
        step = 0
        total_reward = 0
        while not done:
            step += 1
            #greedy method to choose the action
            predictions = q_net(preprocessed_s).detach().numpy()
            a = np.argmax(predictions)
            #take the action
            s_next,reward,done,info = env.step(a)
            total_reward += reward
            #update s 
            preprocessed_s_next = torch.from_numpy(s).unsqueeze(0)
            s = s_next
            preprocessed_s = preprocessed_s_next
        scores = scores + [total_reward]
        n_steps = n_steps +[step]
        #if the past 100 scores have an average of 200 points then break
    return scores, n_steps


#initialize Q and Q target, which are networks with 4 as the number of outputs and 8 as number of inputs
q_net = network(input_size = 8, output_size = 4)
q_net_target = network(input_size = 8, output_size = 4)
q_net.to(device)
q_net_target.to(device)
scores, n_steps, episode, converged = train_dqn(n_episodes = 1000,epsilon = 1,q_net = q_net, q_net_target = q_net_target)
#plot the scores
plt.plot(np.arange(len(scores)),scores)
plt.ylabel('score')
plt.xlabel('ith epsiode')
plt.show()
play_scores, n_steps = play_dqn(q_net = q_net)
#plot the scores
plt.plot(np.arange(len(play_scores)),play_scores)
plt.ylabel('score of a trained agent')
plt.xlabel('ith epsiode')
plt.show()
sum(play_scores)/1000

#train 1000 episodes
#initialize Q and Q target, which are networks with 4 as the number of outputs and 8 as number of inputs
env = gym.make('LunarLander-v2')
q_net_2 = network(input_size = 8, output_size = 4)
q_net_target_2 = network(input_size = 8, output_size = 4)
q_net_2.to(device)
q_net_target_2.to(device)
scores2, n_steps2, episode2, converged2 = train_dqn(n_episodes = 1000,epsilon = 1,q_net = q_net_2, q_net_target = q_net_target_2)
#plot the scores
#rolling_mean = numpy.std(rolling_window(np.array(scores2), 100), 1)
plt.plot(np.arange(len(scores2)),scores2)
#plt.plot(np.arange(len(scores2)), rolling_mean, color='red')
plt.ylabel('score')
plt.xlabel('epsiode')
plt.show()

play_scores2, n_steps2 = play_dqn(q_net = q_net_2)
#plot the scores
plt.plot(np.arange(len(play_scores2)),play_scores2)
plt.ylabel('score of a trained agent,2')
plt.xlabel('ith epsiode')
plt.show()

#hyperparameters
lr_pool = [1e-1,1e-2,1e-3,1e-4,1e-5]
batch_size_pool = [8,16,32,64,128]
gamma_pool = [0.9,0.925,0.95,0.975,0.99]
#method: train on the same number of episodes and 
column_names = ['learning_rate','batch_size','gamma','converged','scores']
tuning = pd.DataFrame(columns = column_names)
n_experiment = 0
for learnr in lr_pool:
    for batch_size in batch_size_pool:
        for gamma in gamma_pool:
            q_net = network(input_size = 8, output_size = 4)
            q_net_target = network(input_size = 8, output_size = 4)
            q_net.to(device)
            q_net_target.to(device)
            print('testing learning rate = ', learnr,'batch_size = ',batch_size, 'gamma=', gamma)
            scores, n_steps, episode, converged = train_dqn(n_episodes = 2000,
                                                 epsilon = 1,
                                                 q_net = q_net, 
                                                 q_net_target = q_net_target,
                                                 lr = learnr,
                                                 gamma = gamma
                                                )
            tuning.loc[n_experiment] = [learnr,batch_size,gamma,converged,str(scores)]
            print(tuning)
            n_experiment += 1
gamma_effect = tuning.query('learning_rate == 0.001 and batch_size == 64')

gamma0900 = np.array(eval(gamma_effect.iloc[0]['scores']))
gamma0900_average = np.convolve(gamma0900, np.ones(100)/100, mode='valid')
gamma0925 = np.array(eval(gamma_effect.iloc[1]['scores']))
gamma0925_average = np.convolve(gamma0925, np.ones(100)/100, mode='valid')
gamma095 = np.array(eval(gamma_effect.iloc[2]['scores']))
gamma095_average = np.convolve(gamma095, np.ones(100)/100, mode='valid')
gamma0975 = np.array(eval(gamma_effect.iloc[3]['scores']))
gamma0975_average = np.convolve(gamma0975, np.ones(100)/100, mode='valid')
gamma099 = np.array(eval(gamma_effect.iloc[4]['scores']))
gamma099_average = np.convolve(gamma099, np.ones(100)/100, mode='valid')

#plot the scores
plt.plot(np.arange(100,1001),gamma0900_average, color ='blue', label = '0.900')
plt.plot(np.arange(100,1001),gamma0925_average, color ='green', label = '0.925')
plt.plot(np.arange(100,1001),gamma095_average, color ='red', label = '0.95')
plt.plot(np.arange(100,1001),gamma0975_average, color ='purple', label = '0.975')
plt.plot(np.arange(100,1001),gamma099_average, color ='skyblue', label = '0.99')
plt.ylabel('score')
plt.xlabel('ith epsiode')
plt.legend()
plt.show()

alpha_effect = tuning.query('gamma == 0.99 and batch_size == 64')

alpha1 = np.array(eval(alpha_effect.iloc[0]['scores'])[0:1000])
alpha1_average = np.convolve(alpha1, np.ones(100)/100, mode='valid')
alpha2 = np.array(eval(alpha_effect.iloc[1]['scores'])[0:1000])
alpha2_average = np.convolve(alpha2, np.ones(100)/100, mode='valid')
alpha3 = np.array(eval(alpha_effect.iloc[2]['scores'])[0:1000])
alpha3_average = np.convolve(alpha3, np.ones(100)/100, mode='valid')
alpha4 = np.array(eval(alpha_effect.iloc[3]['scores'])[0:1000])
alpha4_average = np.convolve(alpha4, np.ones(100)/100, mode='valid')

#plot the scores
plt.plot(np.arange(100,1001),alpha1_average, color ='blue', label = '0.1')
plt.plot(np.arange(100,1001),alpha2_average, color ='green', label = '0.01')
plt.plot(np.arange(100,1001),alpha3_average, color ='red', label = '0.001')
plt.plot(np.arange(100,1001),alpha4_average, color ='purple', label = '0.0001')
plt.ylabel('score')
plt.xlabel('ith epsiode')
plt.legend()
plt.show()

tuning_batch_size = tuning.query('learning_rate == 0.001 and gamma == 0.99')

batch8 = np.array(eval(tuning_batch_size.iloc[0]['scores']))
batch8_average = np.convolve(batch8, np.ones(100)/100, mode='valid')
batch16 = np.array(eval(tuning_batch_size.iloc[1]['scores']))
batch16_average = np.convolve(batch16, np.ones(100)/100, mode='valid')
batch32 = np.array(eval(tuning_batch_size.iloc[2]['scores']))
batch32_average = np.convolve(batch32, np.ones(100)/100, mode='valid')
batch64 = np.array(eval(tuning_batch_size.iloc[3]['scores']))
batch64_average = np.convolve(batch64, np.ones(100)/100, mode='valid')
batch128 = np.array(eval(tuning_batch_size.iloc[4]['scores']))
batch128_average = np.convolve(batch128, np.ones(100)/100, mode='valid')

#plot the scores
plt.plot(np.arange(100,1001),batch8_average, color ='blue', label = '8')
plt.plot(np.arange(100,1001),batch16_average, color ='green', label = '16')
plt.plot(np.arange(100,1001),batch32_average, color ='red', label = '32')
plt.plot(np.arange(100,1001),batch64_average, color ='purple', label = '64')
plt.plot(np.arange(100,1001),batch128_average, color ='skyblue', label = '128')
plt.ylabel('score')
plt.xlabel('ith epsiode')
plt.legend()
plt.show()