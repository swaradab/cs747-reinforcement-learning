'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.02 
        self.epsilon_T2 = 0.05
        self.learning_rate_T1 = 0.075 
        self.learning_rate_T2 = 0.02 
        self.weights_T1 = np.zeros((252*3,))
        self.weights_T2 = np.zeros((3*18*14*4))
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        x_step = 0.1
        v_step = 0.01
        x_states = (self.upper_bounds[0] - self.lower_bounds[0])/x_step
        v_states = (self.upper_bounds[1] - self.lower_bounds[1])/v_step
        state_arr= np.zeros((3, int(x_states*v_states)*3))
        st_id = 3*int(((np.floor((obs[0] - self.lower_bounds[0])/x_step)*v_states) + np.floor((obs[1] - self.lower_bounds[1])/v_step)))
        state_arr[0, st_id] = 1
        state_arr[1, st_id + 1] = 1
        state_arr[2, st_id + 2] = 1
        return state_arr

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''
    def create_tiling(self, start_x, end_x, start_v, end_v, offset_x, offset_v, no_tiles_x, no_tiles_v, tiling_no):
        x = np.linspace(start_x, end_x, no_tiles_x+1) + offset_x*tiling_no
        v = np.linspace(start_v, end_v, no_tiles_v+1) + offset_v*tiling_no
        if tiling_no > 0:
            x[-1], v[-1] = end_x, end_v
        return [x,v]

    def get_better_features(self, obs):
        start_x = self.lower_bounds[0]
        end_x = self.upper_bounds[0]
        start_v = self.lower_bounds[1]
        end_v = self.upper_bounds[1]
        offset_v = 0.0025
        offset_x = 0.025
        no_tiles_x = 18
        no_tiles_v = 14
        no_tilings = round(((end_x - start_x)/no_tiles_x)/offset_x)

        x_tilings = np.empty((int(no_tilings),int(no_tiles_x+1)))
        v_tilings = np.empty((int(no_tilings), int(no_tiles_v+1)))

        for i in range(int(no_tilings)):
            x_tilings[i,:], v_tilings[i,:] = self.create_tiling(start_x, end_x, start_v, end_v, offset_x, offset_v, no_tiles_x, no_tiles_v, i)
        state_arr = np.zeros((3, int(3*no_tiles_x*no_tiles_v*no_tilings)))
        
        for i in range(int(no_tilings)):
            tile_no_x = np.searchsorted(x_tilings[i], obs[0])-1
            tile_no_x = 0 if tile_no_x == -1 else tile_no_x
            tile_no_v = np.searchsorted(v_tilings[i], obs[1])-1
            tile_no_v = 0 if tile_no_v == -1 else tile_no_v
            id_0 = i*(no_tiles_x*no_tiles_v) + (tile_no_x*no_tiles_v) + tile_no_v 
            id_1 = (no_tiles_x*no_tiles_v*no_tilings) + i*(no_tiles_x*no_tiles_v) + (tile_no_x*no_tiles_v) + tile_no_v
            id_2 = (2*no_tiles_x*no_tiles_v*no_tilings) + i*(no_tiles_x*no_tiles_v) + (tile_no_x*no_tiles_v) + tile_no_v
            state_arr[0, id_0] = 1
            state_arr[1, id_1] = 1
            state_arr[2, id_2] = 1
        
        return state_arr

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        x = np.random.choice(2, p=[epsilon, 1-epsilon])
        if x == 0:
            return np.random.choice(3)
        if x == 1:
            return np.argmax(weights@(state.T))

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        
        weights = weights + ((learning_rate*(reward + (self.discount*np.dot(weights, new_state[new_action,:])) - np.dot(weights, state[action,:])))*state[action,:])
        
        if len(weights) == len(self.weights_T1):
            self.weights_T1 = weights
            return self.weights_T1

        if len(weights) == len(self.weights_T2):
            self.weights_T2 = weights
            return self.weights_T2
    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))
