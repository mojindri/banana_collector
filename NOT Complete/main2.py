import os
import pickle
from collections import deque

import torch

from agent import  Agent
import numpy as np
EPS_START  =1.0
EPS_END = 0.1
EPS_DECAY=0.995
import gym

def doTraining(iteration_file, network_file):
    env = gym.make('CliffWalking-v0')

    env_info = env.reset()
    action_sizee = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    state_sizee = len(state)
    agent = Agent(action_size=action_sizee,state_size=state_sizee,use_dueling=True,use_double=True, env = env,  network_file=network_file )
    scores_window = deque(maxlen=100)
    eps = EPS_START
    i=0
    if iteration_file is not None:
        if os.path.exists(iteration_file):
            with open(iteration_file, 'rb') as handle:
                [scores_window, i] = pickle.load(handle)

    for i in range(i, 2000):
        state = env.reset(train_mode=False)[brain_name].vector_observations[0] # reset the environment
        score = 0;
        for t in range(1000):
            action = agent.act(state,eps)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(t,state,next_state,action,reward,done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break
        scores_window.append(score)
        eps = max(EPS_END, eps * EPS_DECAY)
        if i % 5 ==0:

            state = {'local': agent.local_network.state_dict(), 'target':agent.target_network.state_dict(),'optimizer':agent.optimizer.state_dict()}
            torch.save(state, network_file)
            with open(iteration_file, 'wb') as handle:
                pickle.dump([scores_window, i], handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("saved.")
        print("Episode {}, score {}".format(i,np.average(scores_window)))
doTraining('iteration2.pickle','network_3layer2.pth')