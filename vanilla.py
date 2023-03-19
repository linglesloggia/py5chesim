import os
import random
import time
# import tensorflow as tf
from collections import deque

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Conv2D, Dense, Dropout, Embedding,
                          Flatten, MaxPooling2D, Reshape)
# import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model

from tensorflow.keras.optimizers import Adam


from InterSliceSch import InterSliceScheduler
from Slice import *

LEARNING = False  # flag to determine if the agent should be in training mode
MODEL_NAME = "vanilla_v"  # name of the model to be saved
RETRAIN = True  # flag to determine if the model should be trained again or loaded from a saved checkpoint
SHOW_EVERY = 20  # how often to display the current episode number and average score during training

# Defining a new scheduler class that inherits from InterSliceScheduler
class vanillaDQN_Scheduler(InterSliceScheduler):
    def __init__(self, ba, fr, dm, tdd, gr):
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)
        self.granularity = gr  # setting the granularity of the scheduler
        
        # Setting model hyper-parameters
        self.windows_size = 30  # number of time steps to consider in each slice
        self.learning_rate = 0.2  # learning rate for the DQN algorithm
        self.epsilon = 1  # exploration rate (initially set to 100%)
        self.epsilonDecay = 0.99  # rate at which to decay the exploration rate over time
        self.n_slices = 2  # number of slices to divide the network into
        self.n_discrete_states = 20  # number of discrete states per slice
        self.n_actions = 6  # number of possible actions for the agent
        self.lowest_Q_value = -2  # lowest possible Q value
        self.highest_Q_value = 0  # highest possible Q value

        # Setting initial values for the Q-model
        self.episode = 0  # current episode number
        self.discrete_state = [0, 0, 0]  # current discrete state of the environment
        self.new_discrete_state = [0, 0, 0]  # new discrete state after taking an action
        self.SlicesWithPackets = 0  # number of slices with packets
        self.current_state = [0, 0, 0]  # current state of the environment

        # Environment settings
        self.EPISODES = 20_000  # number of episodes to train the agent for
        self.ACTION_SPACE_SIZE = 22  # total number of possible actions (same as self.n_actions)
        self.SLICES = 3  # total number of slices in the network
        self.step = 10  # size of each time step
        self.weights = (  # possible weights for the different slices
            [0,0,0],
            [0, 0, 100],
            [0, 20, 80],
            [0, 40, 60],
            [0, 60, 40],
            [0, 80, 20],
            [0, 100, 0],
            [20, 80, 0],
            [40, 60, 0],
            [60, 40, 0],
            [80, 20, 0],
            [100, 0, 0],
            [80, 0, 20],
            [60, 0, 40],
            [40, 0, 60],
            [20, 0, 80],
            [60, 20, 20],
            [20, 60, 20],
            [20, 20, 60],
            [40, 40, 20],
            [40, 20, 40],
            [20, 40, 40],
        )

        # Exploration settings
        self.epsilon = 1  # the initial value for exploration rate, to be decayed over time
        self.EPSILON_DECAY = 0.7  # the rate at which the exploration rate decays
        self.MIN_EPSILON = 0.001  # the minimum value for the exploration rate
        self.done = 0  # a flag to indicate if the simulation is finished
        self.isRandom = True  # a flag to indicate if the agent is taking random actions

        # Model
        self.agent = DQNAgent()  # initialize a DQNAgent object
        self.current_state = [0, 0, 0]  # initialize the current state of the simulation
        self.div = 1  # a factor used to adjust the simulation time step

        self.action = 0  # the current action taken by the agent
        self.reward = 0  # the current reward earned by the agent

        self.firstMeasuredBytes = [0, 0, 0]  # the number of bytes transmitted in each slice at the beginning of the simulation

        self.hadPackets = []  # a list to keep track of packets that have been transmitted

        def get_reward(self):
            accumulatedReward = 0  # initialize the total accumulated reward
            accumulatedQoE = 0  # initialize the total accumulated QoE
            accumulatedSE = 0  # initialize the total accumulated spectral efficiency
            alpha = 0.5  # a weighting factor for QoE
            beta = 0.5  # a weighting factor for spectral efficiency
            n = 0  # a counter for the number of slices processed
            # Get the reward for each slice
            for slice in list(self.slices.keys()):  # iterate over each slice in the simulation
                nUsers = len(self.slices[slice].schedulerDL.ueLst)  # get the number of users in the slice
                # Check if the slice is idle
                if self.current_state[n] == 0:
                    if self.alloc[n] != 0:
                        accumulatedReward -= 0  # no reward penalty for an idle slice with allocated resources
                # Check if the slice is active
                else:
                    if self.alloc[n] == 0:
                        accumulatedReward -= 0  # no reward penalty for an active slice with no allocated resources
                    else:
                        accumulatedSE += self.set_RBUR(slice)  # add the slice's spectral efficiency to the total
                        accumulatedQoE += self.set_NSRS(slice)  # add the slice's QoE to the total
                n += 1  # increment the slice counter

        accumulatedReward += alpha * accumulatedQoE + beta * accumulatedSE

        # print('Accumulated Reward: ', accumulatedReward)
        # print('Accumulated QoE: ', accumulatedQoE)
        # print('Accumulated SE: ', accumulatedSE)

        with open('Statistics/process/rewardVanilla', 'a') as f:
            f.write(str(accumulatedReward))
            f.write('\n')

        return accumulatedReward

    def set_ue_throughput(self, slice):
        for ue in list(self.slices[slice].schedulerDL.ues.keys()):
            # Calculate the throughput for each user in the slice
            self.slices[slice].schedulerDL.ues[ue].throughput = (
                self.slices[slice].schedulerDL.ues[ue].sndbytes
                * 8000
                / (1024 * 1024 * self.granularity)
            )
            # Reset the sent bytes for the user
            self.slices[slice].schedulerDL.ues[ue].sndbytes = 0

    # Get the RBUR for each slice
    def set_RBUR(self, slice):
        # Get the number of packets transmitted in one second (pks_s) and the total
        # size of all the transmitted packets in one second (tbSize)
        pks_s = self.slices[slice].schedulerDL.pks_s
        tbSize = self.slices[slice].schedulerDL.tbSize

        # Reset the number of transmitted packets and the total size of packets
        self.slices[slice].schedulerDL.pks_s = 0
        self.slices[slice].schedulerDL.tbSize = 0

        # Calculate the RBUR based on the number of transmitted packets and the
        # total size of packets
        if (pks_s > tbSize and tbSize != 0) or pks_s == 0:
            return 1
        elif pks_s != 0 and tbSize == 0:
            return 0
        else:
            return pks_s / tbSize

    # Get the NSRS for each slice
    def set_NSRS(self, slice):
        # Count the number of users that are satisfied with their throughput
        users_satisfied = 0
        n_users = len(self.slices[slice].schedulerDL.ues.keys())

        # Calculate the throughput for each user in the slice
        self.set_ue_throughput(slice)

        # Check if the user's throughput meets the required throughput of the slice
        for ue in list(self.slices[slice].schedulerDL.ues.keys()):
            if (
                self.slices[slice].schedulerDL.ues[ue].throughput
                > self.slices[slice].reqThroughput
            ):
                users_satisfied += 1

        # Calculate the NSRS based on the number of satisfied users and the total number of users
        return users_satisfied / n_users

    def get_weights(self, action):
        # Get the weights for the given action
        return self.weights[action]

    def print_state2(self, state, weights, alloc, every=100):
        # Print the state, action, weights, allocation, and reward for every `every` episodes
        if self.episode % every == 0:
            print("State: ", state)
            print("Action: ", self.action, "Weight: ", weights, "allocation: ", alloc)
            print("Reward: ", self.reward)
            print("---------episeode: --------------------------", self.episode)

    def print_state(self, state, weights, alloc, every=100):
        # Print the state, action, weights, allocation, reward, new state, randomness,
        # and epsilon for every `every` episodes
        if self.episode % every == 0:
            print("State: ", state)
            print("Action: ", self.action, "Weight: ", weights, "allocation: ", alloc)
            print("Reward: ", self.reward)
            print("New State: ", self.new_state)
            print("Random: ", self.isRandom, "Epsilon: ", self.epsilon)
            print("---------episeode: --------------------------", self.episode)

    def resAlloc(self, env):
        if LEARNING:
            if RETRAIN:
                # If RETRAIN is set to True, load the saved model from 'models/{}'.format(MODEL_NAME)
                self.agent.load_model("models/{}".format(MODEL_NAME))

            while True:
                # Write subframe number to a file
                self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")

                # Reset training variables
                self.new_state = []
                nSlice = 0
                alloc = []
                self.alloc = 0
                self.hadPackets = []

                # Choose an action
                if np.random.random() > self.epsilon:
                    # Get the best action from 'Q table'
                    self.action = np.argmax(self.agent.get_qs(self.current_state))
                    bandera = True
                else:
                    # Choose a random action
                    self.action = np.random.randint(0, self.ACTION_SPACE_SIZE)
                    bandera = False

                self.isRandom = bandera

                # Get the set of weights for the previous action
                weights = self.get_weights(self.action)

                # Allocate resources for each slice
                for slice in list(self.slices.keys()):
                    # Calculate the number of PRBs allocated to the slice for the current action
                    prbs = int((self.PRBs * weights[nSlice]) / (100 * self.slices[slice].numRefFactor))

                    # Update the configuration of the slice
                    self.slices[slice].updateConfig(prbs)

                    # Add the number of PRBs allocated to the slice to the allocation list
                    alloc.append(prbs)

                    nSlice += 1

                self.alloc = alloc

                # Update epsilon
                if self.epsilon > self.MIN_EPSILON:
                    self.epsilon *= self.EPSILON_DECAY
                    self.epsilon = max(self.MIN_EPSILON, self.epsilon)

                # Wait for the transition time
                yield env.timeout(self.granularity)

                # Get the system state
                for slice in list(self.slices.keys()):
                    # Compute the number of packets sent in the current subframe by the slice and add it to the state
                    self.new_state.append(min(self.slices[slice].schedulerDL.updSumPcks() // self.div, 500))

                
                # Get the reward
                self.reward = self.get_reward()

                # Update the Q table
                self.agent.update_replay_memory(
                    (
                        self.current_state,
                        self.action,
                        self.reward,
                        self.new_state,
                        self.done,
                    )
                )
                self.agent.train(self.done, self.step)

                self.print_state(self.current_state, weights, alloc, SHOW_EVERY)

                # Update target network counter every episode
                last_state = self.current_state
                self.current_state = self.new_state

                # Update the step
                self.dbFile.write("<hr>")
                self.episode += 1

                # save the model every 1000 episodes

                if self.episode % 20 == 0:
                    self.agent.model.save("models/{}".format(MODEL_NAME))
                    
        if LEARNING == False :
            # If LEARNING is False, predict the action only with the model but not learn.
            # Load the model from the file in 'models/{}'.format(MODEL_NAME)
            self.agent.load_model("models/{}".format(MODEL_NAME))
            
            # Set initial allocation to 0 for each slice
            self.alloc = [0,0,0]

            while True:
                # Write the current subframe number to the database file
                self.dbFile.write("<h3> SUBFRAME NUMBER: " + str(env.now) + "</h3>")
                
                # Reset training variables
                self.current_state = []
                nSlice = 0
                alloc = []

                # Get the system state for each slice
                for slice in list(self.slices.keys()):
                    # Update the state of the slice
                    self.current_state.append(
                        self.slices[slice].schedulerDL.updSumPcks() // self.div
                    )

                # Choose an action based on the predicted Q values from the model
                self.action = np.argmax(self.agent.get_qs(self.current_state)) 

                # Calculate the reward for the current state and action
                self.reward = self.get_reward()

                # Get the set of weights for the previous action
                weights = self.get_weights(self.action)

                # Perform the action chosen for each slice
                for slice in list(self.slices.keys()):
                    # Update the slice configuration with the allocation based on the weights
                    self.slices[slice].updateConfig(
                        (
                            int(
                                (self.PRBs * weights[nSlice])
                                / (100 * self.slices[slice].numRefFactor)
                            )
                        )
                    )
                    # Append the allocation for the slice to the alloc list
                    alloc.append(
                        (
                            int(
                                (self.PRBs * weights[nSlice])
                                / (100 * self.slices[slice].numRefFactor)
                            )
                        )
                    )
                    nSlice += 1

                # Set the allocation for the current state to alloc
                self.alloc = alloc

                # Print the system state, weights, and allocation if SHOW_EVERY conditions are met
                self.print_state2(self.current_state, weights, alloc, SHOW_EVERY)

                # Wait for the transition time
                yield env.timeout(self.granularity)


class DQNAgent:
    def __init__(self):
        self.DISCOUNT = 0.4  # discount factor for future rewards
        self.REPLAY_MEMORY_SIZE = 10  # how many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = 5  # minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = 2  # how many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = 2  # how often to update the target network with main network's weights
        self.MODEL_NAME = "vanilla_v"  # name of the model
        self.MIN_REWARD = -200  # minimum reward for model saving
        self.MEMORY_FRACTION = 0.20  # fraction of GPU memory to use for model training

        self.OBSERVATION_SPACE_VALUES = np.array([[20], [20], [20]])  # shape of the observation space
        self.ACTION_SPACE_SIZE = 22  # size of the action space

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Creates our model
    def create_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=3))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.05), metrics=["mae"])

        return model

    # function that loads the model
    def load_model(self, name):
        self.model = load_model(name)

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (
            current_state,
            action,
            reward,
            new_current_states,
            done,
        ) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.DISCOUNT * max_future_q
            # else:
            #    new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            np.array(X),
            np.array(y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=None,
        )  # [self.tensorboard] if terminal_state else None)

        # Update target network counter every episode

        self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape))[
            0
        ]
