import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import Orthogonal, Zeros
from tqdm import trange
import tensorflow as tf
import time
import argparse

import gym

from Ressim_Env.ressim_enviroment import resSimEnv_v0, resSimEnv_v1


class DQNSolver:

    def __init__(self, observation_space, action_space, MLP_LAYERS, MLP_ACTIVATIONS):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        for layer_n, activation, i in zip(MLP_LAYERS, MLP_ACTIVATIONS, range(len(MLP_LAYERS))):
            if i==0:
                self.model.add(Dense(
                    layer_n, 
                    input_shape=(observation_space,), 
                    activation=activation, 
                    kernel_initializer=Orthogonal(gain=np.sqrt(2.0)), 
                    bias_initializer=Zeros()))
            self.model.add(Dense(
                layer_n, 
                input_shape=(layer_n,), 
                activation=activation,
                kernel_initializer=Orthogonal(gain=np.sqrt(2.0)),
                bias_initializer=Zeros()))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        if USE_TARGET_NETWORK:
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        ### new code for network training
        batch = random.sample(self.memory, BATCH_SIZE)
        state_dim = batch[0][0][0].shape[0] 
        state_np, state_next_np = np.empty((BATCH_SIZE,state_dim)), np.empty((BATCH_SIZE,state_dim))
        reward_np, action_np, done_np = np.empty(BATCH_SIZE), np.empty(BATCH_SIZE), np.empty(BATCH_SIZE)
        for i in range(BATCH_SIZE):
            state_np[i] = (batch[i][0][0])
            state_next_np[i] = (batch[i][3][0])
            action_np[i] = (batch[i][1])
            reward_np[i] = (batch[i][2])
            done_np[i] = (batch[i][4])
        q_t = self.model.predict(state_np)
        if USE_TARGET_NETWORK:
            q_t1 = self.target_model.predict(state_next_np)
        else:
            q_t1 = self.model.predict(state_next_np)
        q_t1_best = np.max(q_t1, axis=1)
        for i in range(BATCH_SIZE):
            q_t[i,int(action_np[i])] = reward_np[i] + GAMMA*(1-done_np[i])*q_t1_best[i]
        # train the DQN network
        self.model.fit(state_np, q_t, verbose=0)

    def eps_timestep_decay(self, t):
        fraction = min (float(t)/int(TOTAL_TIMESTEPS*EXPLORATION_FRACTION), 1.0)
        self.exploration_rate = EXPLORATION_MAX + fraction * (EXPLORATION_MIN - EXPLORATION_MAX)

    def eps_episode_decay(self, episode_num):
        fraction = min (float(episode_num)/EXPLORATION_END_EPISODE, 1.0)
        self.exploration_rate = EXPLORATION_MAX + fraction * (EXPLORATION_MIN - EXPLORATION_MAX)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


def dqn_algorithm(trail_no, verbose=True):

    # for reservoir simulation environemnt
    if ENV_NAME=='ResSim-v0':
        env = resSimEnv_v0(action_steps=ACTION_STEPS, nx=NX,ny=NY,lx=LX,ly=LY,n_steps=N_STEP,dt=DT,mu_w =MU_W,mu_o =MU_O,phi=PHI,k=K,k_type=K_TYPE)
    elif ENV_NAME=='ResSim-v1':
        env = resSimEnv_v1(action_steps=ACTION_STEPS,nx=NX,ny=NY,lx=LX,ly=LY,n_steps=N_STEP,dt=DT,mu_w =MU_W,mu_o =MU_O,phi=PHI,k=K,k_type=K_TYPE)
    else:
        env = gym.make(ENV_NAME) 

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    dqn_solver = DQNSolver(observation_space, action_space, MLP_LAYERS, MLP_ACTIVATIONS)
    t = 0
    episode_rewards = [0.0]
    explore_percent, episodes, mean100_rew, steps = [],[],[],[]
    while True:
        t_record = 0
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            t += 1
            t_record += 1
            #env.render()
            if EXPLORE_DECAY_BY_EPISODES:
                dqn_solver.eps_episode_decay(len(episode_rewards))
            if EXPLORE_DECAY_BY_TIMESTEP:
                dqn_solver.eps_timestep_decay(t)

            action = dqn_solver.act(state)
            state_next, reward, terminal, _ = env.step(action)
            # reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            episode_rewards[-1] += reward
            num_episodes = len(episode_rewards)
            if (terminal or t_record >= STOP_EPISODE_AT_T) and num_episodes%PRINT_FREQ==0:
                explore_percent.append(dqn_solver.exploration_rate*100)
                episodes.append(len(episode_rewards))
                mean100_rew.append(round(np.mean(episode_rewards[(-1-N_EP_AVG):-1]), 1))
                steps.append(t)
                if verbose:
                    print('Exploration %: '+str(int(dqn_solver.exploration_rate*100))+' ,Episodes: '+str(len(episode_rewards))+' ,Mean_reward: '+str(round(np.mean(episode_rewards[(-1-N_EP_AVG):-1]), 1))+' ,timestep: '+str(t))

            if t>TOTAL_TIMESTEPS:
                output_table = np.stack((explore_percent, episodes, mean100_rew, steps))
                if not os.path.exists(FILE_PATH):
                    os.makedirs(FILE_PATH)
                file_name = str(FILE_PATH)+'expt'+str(trail_no)+'.csv'
                np.savetxt(file_name, np.transpose(output_table), delimiter=',', header='Exploration %,Episodes,Rewards,Timestep')
                if SAVE_MODEL:
                    file_name = str(FILE_PATH)+'model'+str(trail_no)+'.h5'
                    dqn_solver.model.save(file_name)
                return
            dqn_solver.experience_replay()
            if USE_TARGET_NETWORK and t%TARGET_UPDATE_FREQUENCY==0:
                dqn_solver.update_target_network()
            if terminal or t_record >= STOP_EPISODE_AT_T:
                episode_rewards.append(0.0)
                break
    return

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # DQN algorithms parameters
    parser.add_argument('--output_folder', default='results/', help='output filepath')
    parser.add_argument('--env_name', default='CartPole-v0', help='string for a gym environment')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total number of timesteps')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='learning rate for the neural network')
    parser.add_argument('--buffer_size',  type=int, default=100, help='Replay buffer size')
    parser.add_argument('--batch_size',  type=int, default=32, help='batch size for experience replay')
    parser.add_argument('--print_frequency',  type=int, default=10, help='results printing episodic frequency')
    parser.add_argument('--exploration_max',  type=float, default=1.0, help='maximum exploration at the begining')
    parser.add_argument('--exploration_min',  type=float, default=0.02, help='minimum exploration at the end')
    parser.add_argument('--exploration_fraction',  type=float, default=0.3, help='fraction of total timesteps on which the exploration decay takes place')
    parser.add_argument('--exploration_end_episode',  type=int, default=150, help='final episode at which exploration value reaches minimum')
    parser.add_argument("--explore_decay_by_timesteps", type=str2bool, default=False,  help="boolean for exploration decay as per timesteps")
    parser.add_argument("--explore_decay_by_episodes", type=str2bool, default=True,  help="boolean for exploration decay as per episodes")
    parser.add_argument('--stop_episode_at_t',  type=int, default=100, help='terminate episode at given timestep')
    parser.add_argument('--n_trial_runs',  type=int, default=1, help='no of trials to run ')
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[64, 64], help='list of neurons in each hodden layer of the DQN network')
    parser.add_argument('--mlp_activations', nargs='+', default=['relu', 'relu'], help='list of activation functions in each hodden layer of the DQN network')
    parser.add_argument("--use_target_network", type=str2bool, default=False,  help="boolean to use target neural network in DQN")
    parser.add_argument('--target_update_frequency',  type=int, default=1, help='timesteps frequency to do weight update from online network to target network')
    parser.add_argument("--verbose", type=str2bool, default=False,  help="print episodic results")
    parser.add_argument('--n_ep_avg',  type=int, default=100, help='no. of episodes to be considered while computing average reward')
    parser.add_argument("--save_model", type=str2bool, default=False,  help="boolean to specify whether the model is to be saved")

    # Reservoir Simulation parameters
    parser.add_argument('--action_steps',  type=int, default=11, help='ResSim parameters: no of actions steps i.e. division of q in given value')
    parser.add_argument('--n_step',  type=int, default=50, help='ResSim parameters: no of steps in one timestep of ressim environment episode ')
    parser.add_argument('--dt',  type=float, default=1e-3, help='ResSim parameters: timestep size of reservoir simulation in each n_step')
    parser.add_argument('--mu_w',  type=float, default=1.0, help='ResSim parameters: water viscosity')
    parser.add_argument('--mu_o',  type=float, default=2.0, help='ResSim parameters: oil viscosity')
    parser.add_argument('--lx',  type=float, default=1.0, help='ResSim parameters: domain length in x direction')
    parser.add_argument('--ly',  type=float, default=1.0, help='ResSim parameters: domain length in y direction')
    parser.add_argument('--nx',  type=int, default=10, help='ResSim parameters: no of cells in x direction ')
    parser.add_argument('--ny',  type=int, default=10, help='ResSim parameters: no of cells in y direction ')
    parser.add_argument('--phi',  type=float, default=0.1, help='ResSim parameters: porosity')
    parser.add_argument('--k',  type=float, default=1.0, help='ResSim parameters: permeability')
    parser.add_argument('--k_type', default='uniform', help='ResSim parameters: permeability type (uniform or random)')
        
    args = parser.parse_args()
    
    # ResSim parameters:
    ACTION_STEPS = args.action_steps 
    N_STEP = args.n_step
    DT = args.dt
    MU_W = args.mu_w
    MU_O = args.mu_o
    LX = args.lx
    LY = args.ly
    NX = args.nx
    NY = args.ny
    PHI = args.phi
    K = args.k
    K_TYPE = args.k_type

    ENV_NAME = args.env_name
    N_EP_AVG = args.n_ep_avg
    SAVE_MODEL = args.save_model
    GAMMA = args.gamma
    LEARNING_RATE = args.learning_rate

    TOTAL_TIMESTEPS = args.total_timesteps
    MEMORY_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_frequency

    EXPLORATION_MAX = args.exploration_max
    EXPLORATION_MIN = args.exploration_min
    EXPLORATION_FRACTION = args.exploration_fraction
    EXPLORATION_END_EPISODE = args.exploration_end_episode
    EXPLORE_DECAY_BY_TIMESTEP = args.explore_decay_by_timesteps
    EXPLORE_DECAY_BY_EPISODES = args.explore_decay_by_episodes

    STOP_EPISODE_AT_T = args.stop_episode_at_t

    FILE_PATH = args.output_folder

    MLP_LAYERS = args.mlp_layers
    MLP_ACTIVATIONS = args.mlp_activations


    N_TRIAL_RUNS = args.n_trial_runs

    USE_TARGET_NETWORK = args.use_target_network
    TARGET_UPDATE_FREQUENCY = args.target_update_frequency

    VERBOSE = args.verbose

    time_array = np.empty(N_TRIAL_RUNS)
    for i in trange(N_TRIAL_RUNS):
        t0 = time.time()
        dqn_algorithm(i, verbose=VERBOSE)
        time_array[i] = time.time() - t0
    np.savetxt(str(FILE_PATH)+'time_taken.csv', time_array, delimiter=',')
