import random
import numpy as np
import os
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import Orthogonal, Zeros
from keras.callbacks import History
from tqdm import trange
import time
import argparse
import gym
from set_seed import set_seed

class DQNSolver:

    def __init__(self, 
                 observation_space, 
                 action_space, 
                 MLP_LAYERS, 
                 MLP_ACTIVATIONS,
                 LEARNING_RATE,
                 EPOCHS,
                 USE_TARGET_NETWORK,
                 GRAD_CLIP,
                 DOUBLE_DQN,
                 LOAD_WEIGHTS,
                 LOAD_WEIGHTS_MODEL_PATH,
                 TOTAL_TIMESTEPS,
                 MEMORY_SIZE,
                 BATCH_SIZE,
                 GAMMA,
                 EXPLORATION_MAX,
                 EXPLORATION_MIN,
                 EXPLORATION_FRACTION):


        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = EXPLORATION_MAX
        self.exploration_max = EXPLORATION_MAX
        self.exploration_min = EXPLORATION_MIN
        self.exploration_fraction = EXPLORATION_FRACTION
        self.mlp_layers = MLP_LAYERS
        self.mlp_activations = MLP_ACTIVATIONS
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.use_target_network = USE_TARGET_NETWORK
        self.grad_clip = GRAD_CLIP
        self.double_dqn =  DOUBLE_DQN
        self.load_weights = LOAD_WEIGHTS
        self.load_weights_model_path = LOAD_WEIGHTS_MODEL_PATH
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.total_timesteps = TOTAL_TIMESTEPS
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.loss = 1.0

        self.model = Sequential()
        for layer_n, activation, i in zip(self.mlp_layers, self.mlp_activations, range(len(self.mlp_layers))):
            if i==0:
                self.model.add(Dense(
                    layer_n, 
                    input_shape=(self.observation_space,), 
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
        if self.load_weights:
            self.model.load_weights(self.load_weights_model_path)
        if self.grad_clip:
            self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate, amsgrad=True, clipvalue=10.0))
        else:
            self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate, amsgrad=True))
        if self.use_target_network:
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        ### new code for network training
        batch = random.sample(self.memory, self.batch_size)
        state_dim = batch[0][0][0].shape[0] 
        state_np, state_next_np = np.empty((self.batch_size,state_dim)), np.empty((self.batch_size,state_dim))
        reward_np, action_np, done_np = np.empty(self.batch_size), np.empty(self.batch_size), np.empty(self.batch_size)
        for i in range(self.batch_size):
            state_np[i] = (batch[i][0][0])
            state_next_np[i] = (batch[i][3][0])
            action_np[i] = (batch[i][1])
            reward_np[i] = (batch[i][2])
            done_np[i] = (batch[i][4])
        q_t = self.model.predict(state_np)
        if self.use_target_network:
            q_t1 = self.target_model.predict(state_next_np)
        else:
            q_t1 = self.model.predict(state_next_np)
        q_t1_best = np.max(q_t1, axis=1)
        if self.double_dqn and self.use_target_network:
            q_t1_local = self.model.predict(state_next_np)
            ind = np.argmax(q_t1_local, axis=1)
        for i in range(self.batch_size):
            if self.double_dqn and self.use_target_network:
                q_t1_best[i] = q_t1[i,ind[i]]
            q_t[i,int(action_np[i])] = reward_np[i] + self.gamma*(1-done_np[i])*q_t1_best[i]
        # train the DQN network
        history = History()
        hist = self.model.fit(state_np, q_t, verbose=0, epochs=self.epochs, callbacks=[history])
        self.loss=hist.history['loss'][-1]

    def eps_timestep_decay(self, t):
        fraction = min (float(t)/int(self.total_timesteps*self.exploration_fraction), 1.0)
        self.exploration_rate = self.exploration_max + fraction * (self.exploration_min - self.exploration_max)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


def dqn_algorithm(ENV_NAME,
                  SEED=1,
                  TOTAL_TIMESTEPS = 100000,
                  GAMMA = 0.95,
                  MEMORY_SIZE = 1000,
                  BATCH_SIZE = 32,
                  EXPLORATION_MAX = 1.0,
                  EXPLORATION_MIN = 0.02,
                  EXPLORATION_FRACTION = 0.7,
                  TRAINING_FREQUENCY = 1000,
                  FILE_PATH = 'results/',
                  SAVE_MODEL = False,
                  MODEL_FILE_NAME = 'model',
                  LOG_FILE_NAME = 'log',
                  TIME_FILE_NAME = 'time',
                  PRINT_FREQ = 100,
                  N_EP_AVG = 100,
                  VERBOSE = 'False',
                  MLP_LAYERS = [64,64],
                  MLP_ACTIVATIONS = ['relu','relu'],
                  LEARNING_RATE = 1e-3,
                  EPOCHS = 1,
                  GRAD_CLIP = False,
                  DOUBLE_DQN = False,
                  USE_TARGET_NETWORK = True,
                  TARGET_UPDATE_FREQUENCY = 5000,
                  LOAD_WEIGHTS = False,
                  LOAD_WEIGHTS_MODEL_PATH = 'results/model0.h5'):

    '''
    DQN Algorithm execution

    env_name : string for a gym environment
    total_timesteps : Total number of timesteps
    gamma : discount factor : 
    buffer_size : Replay buffer size 
    batch_size : batch size for experience replay 
    exploration_max : maximum exploration at the begining 
    exploration_min : minimum exploration at the end 
    exploration_fraction : fraction of total timesteps on which the exploration decay takes place 
    output_folder : output filepath 
    save_model : boolean to specify whether the model is to be saved 
    model_file_name : name of file to save the model at the end learning 
    log_file_name : name of file to store DQN results 
    time_file_name : name of file to store computation time 
    print_frequency : results printing episodic frequency 
    n_ep_avg : no. of episodes to be considered while computing average reward 
    verbose : print episodic results 
    mlp_layers : list of neurons in each hodden layer of the DQN network 
    mlp_activations : list of activation functions in each hodden layer of the DQN network 
    learning_rate : learning rate for the neural network 
    epochs : no. of epochs in every experience replay 
    grad_clip : boolean to specify whether to use gradient clipping in the optimizer (graclip value 10.0) 
    double_dqn : boolean to specify whether to employ double DQN 
    use_target_network : boolean to use target neural network in DQN 
    target_update_frequency : timesteps frequency to do weight update from online network to target network 
    load_weights : boolean to specify whether to use a prespecified model to initializa the weights of neural network 
    load_weights_model_path : path for the model to use for weight initialization 
    '''

    before = time.time()
    env = gym.make(ENV_NAME)

    # for reproducibility
    env.seed(SEED)
    set_seed(SEED)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    dqn_solver = DQNSolver(observation_space, 
                           action_space, 
                           MLP_LAYERS, 
                           MLP_ACTIVATIONS,
                           LEARNING_RATE,
                           EPOCHS,
                           USE_TARGET_NETWORK,
                           GRAD_CLIP,
                           DOUBLE_DQN,
                           LOAD_WEIGHTS,
                           LOAD_WEIGHTS_MODEL_PATH,
                           TOTAL_TIMESTEPS,
                           MEMORY_SIZE,
                           BATCH_SIZE,
                           GAMMA,
                           EXPLORATION_MAX,
                           EXPLORATION_MIN,
                           EXPLORATION_FRACTION)
    t = 0
    episode_rewards = [0.0]
    explore_percent, episodes, mean100_rew, steps, NN_tr_loss = [],[],[],[],[]
    while True:
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            t += 1
            dqn_solver.eps_timestep_decay(t)

            action = dqn_solver.act(state)
            state_next, reward, terminal, _ = env.step(action)
            # reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            if t%TRAINING_FREQUENCY==0:
                dqn_solver.experience_replay()
            state = state_next
            episode_rewards[-1] += reward
            num_episodes = len(episode_rewards)
            if (terminal and num_episodes%PRINT_FREQ==0):
                explore_percent.append(dqn_solver.exploration_rate*100)
                episodes.append(len(episode_rewards))
                mean100_rew.append(round(np.mean(episode_rewards[(-1-N_EP_AVG):-1]), 1))
                steps.append(t)
                NN_tr_loss.append(dqn_solver.loss)
                if VERBOSE:
                    print('Exploration %: '+str(int(explore_percent[-1]))+' ,Episodes: '+str(episodes[-1])+' ,Mean_reward: '+str(mean100_rew[-1])+' ,timestep: '+str(t)+' , tr_loss: '+str(round(NN_tr_loss[-1],4)))

            if t>TOTAL_TIMESTEPS:
                output_table = np.stack((explore_percent, episodes, mean100_rew, steps, NN_tr_loss))
                if not os.path.exists(FILE_PATH):
                    os.makedirs(FILE_PATH)
                file_name = str(FILE_PATH)+LOG_FILE_NAME+'.csv'
                np.savetxt(file_name, np.transpose(output_table), delimiter=',', header='Exploration %,Episodes,Rewards,Timestep,Training Score')
                after = time.time()
                time_taken = after-before
                np.save( str(FILE_PATH)+TIME_FILE_NAME, time_taken )
                if SAVE_MODEL:
                    file_name = str(FILE_PATH)+MODEL_FILE_NAME+'.h5'
                    dqn_solver.model.save(file_name)
                return
            if USE_TARGET_NETWORK and t%TARGET_UPDATE_FREQUENCY==0:
                dqn_solver.update_target_network()
            if terminal:
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
    parser.add_argument('--env_name', default='CartPole-v0', help='string for a gym environment')
    parser.add_argument('--seed', type=int, default=4, help='seed for pseudo random generator')
    parser.add_argument('--total_timesteps', type=int, default=250000, help='Total number of timesteps')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_size',  type=int, default=1000, help='Replay buffer size')
    parser.add_argument('--batch_size',  type=int, default=32, help='batch size for experience replay')
    parser.add_argument('--exploration_max',  type=float, default=1.0, help='maximum exploration at the begining')
    parser.add_argument('--exploration_min',  type=float, default=0.02, help='minimum exploration at the end')
    parser.add_argument('--exploration_fraction',  type=float, default=0.6, help='fraction of total timesteps on which the exploration decay takes place')
    parser.add_argument('--output_folder', default='results/', help='output filepath')
    parser.add_argument('--save_model', type=str2bool, default=False,  help='boolean to specify whether the model is to be saved')
    parser.add_argument('--model_file_name', default='model', help='name of file to save the model at the end learning')
    parser.add_argument('--log_file_name', default='log', help='name of file to store DQN results')
    parser.add_argument('--time_file_name', default='time', help='name of file to store computation time')
    parser.add_argument('--print_frequency',  type=int, default=100, help='results printing episodic frequency')
    parser.add_argument('--n_ep_avg',  type=int, default=100, help='no. of episodes to be considered while computing average reward')
    parser.add_argument('--verbose', type=str2bool, default=True,  help='print episodic results')
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[64, 64], help='list of neurons in each hodden layer of the DQN network')
    parser.add_argument('--mlp_activations', nargs='+', default=['relu', 'relu'], help='list of activation functions in each hodden layer of the DQN network')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='learning rate for the neural network')
    parser.add_argument('--epochs',  type=int, default=1, help='no. of epochs in every experience replay')
    parser.add_argument('--grad_clip', type=str2bool, default=False,  help='boolean to specify whether to use gradient clipping in the optimizer (graclip value 10.0)')
    parser.add_argument('--double_dqn', type=str2bool, default=False,  help='boolean to specify whether to employ double DQN')
    parser.add_argument('--use_target_network', type=str2bool, default=True,  help='boolean to use target neural network in DQN')
    parser.add_argument('--target_update_frequency',  type=int, default=1000, help='timesteps frequency to do weight update from online network to target network')
    parser.add_argument('--training_frequency',  type=int, default=100, help='timesteps frequency to train the DQN (experience replay)')
    parser.add_argument('--load_weights", type=str2bool, default=False,  help="boolean to specify whether to use a prespecified model to initializa the weights of neural network')
    parser.add_argument('--load_weights_model_path', default='results/model0.h5', help='path for the model to use for weight initialization')
    args = parser.parse_args()
    
    '''
    # List of parameters

    # DQN algorithm parameters
    ENV_NAME = args.env_name
    GAMMA = args.gamma
    TOTAL_TIMESTEPS = args.total_timesteps
    MEMORY_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    TARGET_UPDATE_FREQUENCY = args.target_update_frequency

    # saving/loggin parameters
    PRINT_FREQ = args.print_frequency
    N_EP_AVG = args.n_ep_avg
    SAVE_MODEL = args.save_model
    FILE_PATH = args.output_folder
    MODEL_FILE_NAME = args.model_file_name
    LOG_FILE_NAME = args.log_file_name
    TIME_FILE_NAME = args.time_file_name
    VERBOSE = args.verbose

    # DQNSolver parameters
    EPOCHS = args.epochs
    GRAD_CLIP = args.grad_clip
    MLP_LAYERS = args.mlp_layers
    MLP_ACTIVATIONS = args.mlp_activations
    DOUBLE_DQN = args.double_dqn
    LEARNING_RATE = args.learning_rate
    EXPLORATION_MAX = args.exploration_max
    EXPLORATION_MIN = args.exploration_min
    EXPLORATION_FRACTION = args.exploration_fraction
    USE_TARGET_NETWORK = args.use_target_network
    LOAD_WEIGHTS = args.load_weights
    LOAD_WEIGHTS_MODEL_PATH = args.load_weights_model_path
    '''

    dqn_algorithm(ENV_NAME=args.env_name,
                  SEED=args.seed,
                  GAMMA = args.gamma,
                  TOTAL_TIMESTEPS = args.total_timesteps,
                  MEMORY_SIZE = args.buffer_size,
                  BATCH_SIZE = args.batch_size,
                  TRAINING_FREQUENCY = args.training_frequency,
                  TARGET_UPDATE_FREQUENCY = args.target_update_frequency,
                  PRINT_FREQ = args.print_frequency,
                  N_EP_AVG = args.n_ep_avg,
                  SAVE_MODEL = args.save_model,
                  FILE_PATH = args.output_folder,
                  MODEL_FILE_NAME = args.model_file_name,
                  LOG_FILE_NAME = args.log_file_name,
                  TIME_FILE_NAME = args.time_file_name,
                  VERBOSE = args.verbose,
                  EPOCHS = args.epochs,
                  GRAD_CLIP = args.grad_clip,
                  MLP_LAYERS = args.mlp_layers,
                  MLP_ACTIVATIONS = args.mlp_activations,
                  DOUBLE_DQN = args.double_dqn,
                  LEARNING_RATE = args.learning_rate,
                  EXPLORATION_MAX = args.exploration_max,
                  EXPLORATION_MIN = args.exploration_min,
                  EXPLORATION_FRACTION = args.exploration_fraction,
                  USE_TARGET_NETWORK = args.use_target_network,
                  LOAD_WEIGHTS = args.load_weights,
                  LOAD_WEIGHTS_MODEL_PATH = args.load_weights_model_path)
