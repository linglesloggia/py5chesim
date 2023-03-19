# InterSliceScheduler using APEX
# This is an adaptation of the work in Radio Resource Allocation Method for Network Slicing using
# Deep Reinforcement Learning from Yu Abiko, Takato Saito, Daizo Ikeda, Ken Ohta, 
# Tadanori Mizuno, and Hiroshi Mineno.
# State:    (NSRS, RBUR, nRB, reqThroughput, ReqDelay, nUE, ueTraffic)
# Action:   IDRB = floor(     -1^(act) . 2^(florr(act/2)-1)    )
#           ARB_t = ART_(t-1) + IDRB
# Reward:   NSRS x RBUR

import math
import time
from collections import deque
from decimal import ROUND_HALF_UP, Decimal, getcontext

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Dropout, Embedding,
                          Flatten, MaxPooling2D, Reshape)
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from InterSliceSch import InterSliceScheduler
from Slice import *

# Set global variables
MODEL_NAME = "apex_v3"
FOLDER_NAME = "models_v3"
MODEL_NAME_TF = "256x2"

# Set learning parameters
LEARNING = False
RETRAIN = False
tSim = 6_000 # Simulation time
SAVE_EVERY = 100 # Number of episodes before saving the model

# Set agent parameters
REPLAY_MEMORY_SIZE = 50_000 # Maximum size of the replay buffer
MIN_REPLAY_MEMORY_SIZE = 100 # Minimum size of the replay buffer

ACTION_SPACE_SIZE = 8 # Number of actions the agent can take
DISCOUNT = 0.3 # Discount factor used in the Q-learning update rule
MINIBATCH_SIZE = 64 # Size of the minibatch used in the Q-learning update rule
UPDATE_TARGET_EVERY = 200 # Number of episodes before updating the target network
SHOW_EVERY = 20 # Number of episodes before showing the current Q-values
EPSILON_DECAY_RATE = 0.95 # Epsilon decay rate used in the epsilon-greedy policy
EPSILON_DECAY_RATE_TRAINED = 0.95 # Epsilon decay rate used after the agent is trained

# Define the Apex_Scheduler class
class Apex_Scheduler(InterSliceScheduler):
    def __init__(self, ba, fr, dm, tdd, gr):
        InterSliceScheduler.__init__(self, ba, fr, dm, tdd, gr)
        self.new_state = [] # List containing the new state of the environment
        self.replayBuffer = [] # List containing the experience replay buffer
        self.agent = DQNAgent() # Create a DQNAgent object
        self.prbBuffer = 0 # Number of PRBs allocated to the current slice
        self.epsilon = 1 # Epsilon value used in the epsilon-greedy policy
        self.episode = 0 # Current episode number
        self.totalPrbs = 52 # Total number of PRBs in the system
        self.prbBuffer = 52 # Number of PRBs allocated to the current slice
        self.granularity = gr # Granularity of the simulation

    def resAlloc(self, env):
        """This method is the main method of the scheduler. It is responsible for the resource allocation process."""
        if LEARNING:
            if RETRAIN:
                # If RETRAIN is True, load the model from the file in 'models/{}'.format(MODEL_NAME)
                self.agent.load_model("models/{}".format(MODEL_NAME))
        
            current_states = []
            self.is_random = False
            while env.now < (tSim * 0.83):
                actions = []

                for slice in list(self.slices.keys()):
                    # If the slice is not BestEffort
                    if self.slices[slice].label != "BestEffort":
                        # Choose an action using the epsilon-greedy policy
                        if np.random.random() > self.epsilon:
                            action = np.argmax(
                                self.agent.get_qs(np.array(self.slices[slice].state))
                            )
                            self.is_random = False
                        else:
                            action = np.random.randint(0, ACTION_SPACE_SIZE)
                            self.is_random = True

                        # Decay the value of epsilon
                        self.decay_epsilon()

                        # Assign PRBs to the slice based on the chosen action

                    else:
                        # If the slice is BestEffort
                        self.slices[slice].updateConfig((self.prbBuffer))
                    
                    

                    actions.append(action)

                # This line yields a delay until the next granularity time.
                yield env.timeout(self.granularity)

                # This loop iterates over each slice in the dictionary of slices.
                for slice in list(self.slices.keys()):

                    # This block of code updates the state of the slice.
                    # The state is a list of values representing the state of the slice.
                    # Each value corresponds to a different aspect of the slice's state.
                    # These values are set using various methods of the slice object.
                    # The values are discretized and stored as integers.
                    NSRS = self.set_NSRS(slice)
                    RBUR = self.set_RBUR(slice)
                    ueTraffic = self.slices[slice].schedulerDL.updSumPcks()

                    discreteNSRS = self.discretize(NSRS, 0, 1)
                    discreteRBUR = self.discretize(RBUR, 0, 1)
                    discrete_ueTraffic = self.discretize2(ueTraffic, 0, 1_000, 10)

                    self.slices[slice].NSRS = discreteNSRS
                    self.slices[slice].RBUR = discreteRBUR
                    self.slices[slice].ueTraffic = discrete_ueTraffic

                    self.slices[slice].nUE = len(self.slices[slice].schedulerDL.ueLst)
                    self.slices[slice].ueTraffic = self.slices[
                        slice
                    ].schedulerDL.updSumPcks()

                    self.set_ue_throughput(slice)

                    if self.slices[slice].label != "BestEffort":
                        self.slices[slice].state = [
                            self.slices[slice].NSRS,
                            self.slices[slice].RBUR,
                            self.slices[slice].PRBs,
                            self.slices[slice].reqThroughput,
                            self.slices[slice].reqDelay,
                            self.slices[slice].nUE,
                            self.slices[slice].ueTraffic,
                        ]

                        new_states.append(self.slices[slice].state)

                        self.slices[slice].reward = self.get_reward(slice)
                        rewards.append(self.slices[slice].reward)
                

                if self.episode >= 5:
                    self.update_replayBuffer(
                        current_states, actions, rewards, new_states
                    )
                    self.agent.train(self.replayBuffer)

                current_states = new_states

                self.episode = self.episode + 1
                # self.printSliceConfig(slice)

                self.print_env()

                if self.episode % SAVE_EVERY == 0:
                    self.agent.save_model("models/{}".format(MODEL_NAME))
            # End of training
            # self.create_folder(FOLDER_NAME)
            # self.save_model(agent.model, MODEL_NAME)


        else:
            # Load the previously saved model
            self.agent.load_model("models/{}".format(MODEL_NAME))

            # Create an empty list to hold current states
            current_states = []
            self.is_random = False

            # If file exists, delete it
            if os.path.exists('Statistics/process/apexdata'):
                os.remove('Statistics/process/apexdata')

            # Set some initial values
            first = True
            nSlc = 0

            # Run simulation until 80% of total simulation time
            while env.now < 0.8 * tSim:
                # Write subframe number to database file
                self.dbFile.write('<h3> SUBFRAME NUMBER: '+str(env.now)+'</h3>')

                # Loop through slices and perform actions
                for slice in list(self.slices.keys()):
                    # If slice is not a BestEffort slice
                    if self.slices[slice].label != "BestEffort":
                        # If not the first iteration, choose action based on the maximum predicted Q-value
                        if not first:
                            # Choose action with highest predicted Q-value for current state
                            action = np.argmax(
                                self.agent.get_qs(np.array(self.slices[slice].state))
                            )
                            self.is_random = False
                            
                        # If first iteration, choose a random action
                        else:
                            # Choose a random action
                            action = np.random.randint(0, ACTION_SPACE_SIZE)
                            self.is_random = True
                            nSlc += 1

                            # If already chosen 2 random actions, switch to choosing based on Q-values
                            if nSlc > 2:
                                first = False

                        if self.epsilon >= 0.01:
                            self.epsilon = EPSILON_DECAY_RATE_TRAINED*self.epsilon

                        self.assign_prbs(action, slice)

                    else:
                        self.slices[slice].updateConfig((self.prbBuffer))

                self.dbFile.write("<hr>")

                yield env.timeout(self.granularity)

                new_states = []
                rewards = []

                for slice in list(self.slices.keys()):
                    # State:    (NSRS, RBUR, nRB, reqThroughput, ReqDelay, nUE, ueTraffic)

                    # By Slice, amount of:
                    #                       PRBS,
                    #                       UE,
                    #                       packets in bearers,
                    #                       Thgpt requirements,
                    #                       Delay requirements,
                    #                       NSRS: requirements satisfaction indicator,
                    #                       RBUR: Resource Block Usage Ratio.
                    #
                    # self.slices[slice].reqThroughput is already updated from the start
                    # self.slices[slice].reqDelay is already updated from the start
                    # self.slices[slice].NSRS = self.slices[slice].schedulerDL.NSRS
                    # self.slices[slice].RBUR = (self.slices[slice].schedulerDL.PRBs_used/self.slices[slice].PRBs if self.slices[slice].PRBs != 0 else 0)

                    NSRS = self.set_NSRS(slice)
                    RBUR = self.set_RBUR(slice)
                    ueTraffic = self.slices[slice].schedulerDL.updSumPcks()

                    # self.discretize(self, value, min, max, granularity):
                    discreteNSRS = self.discretize(NSRS, 0, 1)
                    discreteRBUR = self.discretize(RBUR, 0, 1)
                    discrete_ueTraffic = self.discretize2(ueTraffic, 0, 1_000, 10)

                    self.slices[slice].NSRS = discreteNSRS
                    self.slices[slice].RBUR = discreteRBUR
                    self.slices[slice].ueTraffic = discrete_ueTraffic

                    self.slices[slice].nUE = len(self.slices[slice].schedulerDL.ueLst)
                    self.slices[slice].ueTraffic = self.slices[
                        slice
                    ].schedulerDL.updSumPcks()
                    # print(self.slices[slice].RBUR, self.slices[slice].NSRS, 'RBUR, NSRS')
                    self.set_ue_throughput(slice)
                    self.slices[slice].schedulerDL.show = False

                    if self.slices[slice].label != "BestEffort":
                        self.slices[slice].state = [
                            self.slices[slice].NSRS,
                            self.slices[slice].RBUR,
                            self.slices[slice].PRBs,
                            self.slices[slice].reqThroughput,
                            self.slices[slice].reqDelay,
                            self.slices[slice].nUE,
                            self.slices[slice].ueTraffic,
                        ]
                        new_states.append(self.slices[slice].state)
                        self.slices[slice].reward = self.get_reward(slice)
                        
                        to_save = [self.slices[slice].label, self.slices[slice].reward, self.slices[slice].state[0], self.slices[slice].state[1],self.slices[slice].state[6]]

                        # save to file
                        self.save_to_file(to_save)
                        self.save_to_file2(env.now)

                current_states = new_states

                self.episode = self.episode + 1
                # self.printSliceConfig(slice)

                
                #self.printSliceConfig(slice)
                self.print_env()

    # Normalize the input value in ten possible values
    def normalize(self, value, bins):
        return self.discretize(value, bins) / (len(bins) + 1)
    
    def save_to_file2(self, to_save):
        with open('Statistics/process/timestamp', 'a') as f:
            f.write(str(to_save))
            f.write('\n')
    # fucntion that appends an input to a file
    def save_to_file(self, to_save):
        with open('Statistics/process/apexdata', 'a') as f:
            f.write(str(to_save))
            #f.write('\n')

    # discretizes the input into 10 possible values
    def discretize2(self, val, min_val, max_val, div):
        val = round((val - min_val) / ((max_val - min_val) * div), 1)
        val = np.clip(val, min_val, max_val)
        return round(val, 1)

    def discretize(self, val, min_val, max_val):
        val = np.round((val - min_val) / (max_val - min_val), 1)
        # val = np.clip(val, min_val, max_val)
        return val  # round(val,1)

    # other function that bins the input into 10 possible values

    def discretize4(self, val, min_val, max_val, granularity):
        granularity = Decimal(granularity)
        val = Decimal(val)
        getcontext().prec = 2
        val = getcontext().quantize(val, granularity)
        val = min(max(val, Decimal(min_val)), Decimal(max_val))
        return float(val)


    # Load a model with the name of the model
    def load_model(self, name):
        self.agent.model = load_model(f"models/{name}.model")

    # Create a folder if it doesn't exist
    def create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Set the throughput value for each ue
    def set_ue_throughput(self, slice):
        for ue in list(self.slices[slice].schedulerDL.ues.keys()):
            self.slices[slice].schedulerDL.ues[ue].throughput = (
                self.slices[slice].schedulerDL.ues[ue].sndbytes
                * 8000
                / (1024 * 1024 * self.granularity)
            )
            self.slices[slice].schedulerDL.ues[ue].sndbytes = 0

    # Save the list in separated lines of a file
    def save_list(self, list, name):
        with open(name, "w") as f:
            for item in list:
                f.write("%s " % item)

    # append the list in separated lines of a file
    def append_list(self, list, name):
        with open(name, "a") as f:
            for item in list:
                f.write("%s \n" % item)

    # Get the RBUR for each slice
    def set_RBUR(self, slice):
        # for ue in list(self.slices[slice].schedulerDL.ues.keys()):
        # print('gianver: ', self.slices[slice].schedulerDL.ues[ue].pks_s / self.slices[slice].schedulerDL.ues[ue].tbSize)
        pks_s = self.slices[slice].schedulerDL.pks_s
        tbSize = self.slices[slice].schedulerDL.tbSize

        self.slices[slice].schedulerDL.pks_s = 0
        self.slices[slice].schedulerDL.tbSize = 0

        if (pks_s > tbSize and tbSize != 0) or pks_s == 0:
            return 1
        elif pks_s != 0 and tbSize == 0:
            return 0
        else:
            return pks_s / tbSize

        # self.slices[slice].schedulerDL.ues[ue].RBUR = self.slices[slice].schedulerDL.ues[ue].pks_s / self.slices[slice].schedulerDL.ues[ue].tbSize

    # Get the NSRS for each slice
    def set_NSRS(self, slice):
        users_satisfied = 0
        n_users = len(self.slices[slice].schedulerDL.ues.keys())
        #print(self.slices[slice].reqThroughput, 'slice: ', self.slices[slice].label)
        for ue in list(self.slices[slice].schedulerDL.ues.keys()):
            if (
                self.slices[slice].schedulerDL.ues[ue].throughput
                > self.slices[slice].reqThroughput
            ):
                # if slice == 'eMBB-1':
                #    print(self.slices[slice].schedulerDL.ues[ue].throughput, self.slices[slice].reqThroughput)
                users_satisfied += 1

        # print('user_satisfied, nuser_ cociente: ',users_satisfied, n_users, users_satisfied/n_users)
        if n_users == 0:
            return 0
        else:
            return users_satisfied / n_users

    # Creates tensorboard files to be used for visualization
    def setup_tensorboard(self):
        # Name for tensorboard
        NAME = f"{MODEL_NAME_TF}-{int(time.time())}"
        tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
        return tensorboard

    # prints periodically the environment setup
    def print_env(self):
        if self.episode % SHOW_EVERY == 0:
            for slice in list(self.slices.keys()):
                print(
                    "\nSlice: ",
                    self.slices[slice].label,
                    #self.slices[slice].schedulerDL.PRBs_used,
                    self.slices[slice].PRBs,
                    "\nPRBs: ",
                    self.slices[slice].PRBs,
                    ", Epsilon: ",
                    self.epsilon,
                    ", Random: ",
                    self.is_random,
                    "\nReward: ",
                    self.slices[slice].reward,
                    ", NSRS: ",
                    self.slices[slice].NSRS,
                    ", RBUR: ",
                    self.slices[slice].RBUR,
                )
                self.slices[slice].schedulerDL.show = True

    def decay_epsilon(self):
        self.epsilon = EPSILON_DECAY_RATE * self.epsilon

    def get_reward(self, slice):
        
        if self.slices[slice].ueTraffic != 0 and self.slices[slice].PRBs == 0:
            reward = 0
        elif self.slices[slice].ueTraffic == 0 and self.slices[slice].PRBs == 0:
            reward = 1
        elif self.slices[slice].ueTraffic == 0 and self.slices[slice].PRBs != 0:
            reward = 0
        else:
            reward = self.slices[slice].NSRS * self.slices[slice].RBUR

        #if self.slices[slice].buffer_availability == 0:
        #    reward = -0.1
        return reward

    def assign_prbs(self, act, slice):
        IDRB = np.floor(pow(-1, act) * pow(2, np.floor(act / 2) - 1))

        if self.slices[slice].PRBs + IDRB <= 0:
            IDRB = -self.slices[slice].PRBs

        elif self.slices[slice].PRBs + IDRB >= self.PRBs:
            IDRB = -self.PRBs + self.slices[slice].PRBs

        if self.prbBuffer - IDRB <= 0:
            PRBs = self.slices[slice].PRBs + self.prbBuffer
            self.prbBuffer = 0

        elif self.prbBuffer - IDRB >= self.PRBs:
            PRBs = 0
            self.prbBuffer = self.PRBs

        else:
            PRBs = self.slices[slice].PRBs + IDRB
            self.prbBuffer = self.prbBuffer - IDRB

        if self.prbBuffer == 0:
            self.slices[slice].buffer_available == False
        else:
            self.slices[slice].buffer_available == True

        """
        if (self.prbBuffer - IDRB >= 0) and (self.prbBuffer - IDRB <= self.PRBs):
            PRBs = (self.slices[slice].PRBs + IDRB) 
            self.prbBuffer = self.prbBuffer - IDRB
        """

        # print('Slice: ', self.slices[slice].label, 'PRBs: ', PRBs, 'buffer: ', self.prbBuffer, 'suma: ', PRBs + self.prbBuffer)
        self.slices[slice].updateConfig(PRBs)
        self.slices[slice].PRBs = PRBs

        """
        PRBs = (self.slices[slice].PRBs + IDRB) if (self.prbBuffer - IDRB >= 0) \
                            and (self.prbBuffer - IDRB <= self.PRBs) \
                            else (self.slices[slice].PRBs + self.prbBuffer if self.prbBuffer - IDRB < 0 \
                            else 0)

        self.prbBuffer = self.prbBuffer - IDRB if (self.prbBuffer - IDRB >= 0) \
                            and (self.prbBuffer - IDRB <= self.PRBs) \
                            else (0 if self.prbBuffer - IDRB < 0 \
                            else self.PRBs)"""

    def update_replayBuffer(self, current_states, actions, rewards, new_states):
        transition = Transition()

        for i in range(len(current_states)):
            transition.current_states.append(current_states[i])
            transition.actions.append(actions[i])
            transition.rewards.append(rewards[i])
            transition.new_states.append(new_states[i])

        self.replayBuffer.append(transition)


class Transition:
    def __init__(self):
        self.current_states = []
        self.actions = []
        self.rewards = []
        self.new_states = []


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, "train")
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class ModifiedTensorBoard2(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(MODEL_NAME_TF)
        self._log_write_dir = MODEL_NAME_TF

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, "train")
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


# Deep Q-learning Agent

class DQNAgent:
    def __init__(self):
        # Initialize the main model
        self.model = self._build_model()
        # Initialize the target network with the same model as the main model
        self.target_model = self._build_model()
        # Set the weights of the target model to be the same as the main model
        self.target_model.set_weights(self.model.get_weights())
        # Create a tensorboard object for logging
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME_TF, int(time.time()))
        )
        # Initialize a debug counter for tracking
        self.debug_counter = 0

        # Create a replay memory deque to store the last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Build the neural network model
    def _build_model(self):
        model = Sequential()
        # Add layers to the model with the specified activations
        model.add(Dense(64, activation="relu", input_dim=7))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(ACTION_SPACE_SIZE, activation="linear"))
        # Compile the model with a mean squared error loss function and Adam optimizer
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])

        return model

    # Save the model to file
    def save_model(self,name):
        self.model.save(name)

    # Load the model from file
    def load_model(self, name):
        self.model = load_model(name)

    # Get the predicted Q-values for a given state
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    # Train the neural network on the provided replay memory
    def train(self, replay_memory):
        # If there is not enough data in the replay memory, do not train
        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get the current states, new states, rewards, and actions from the replay memory
        current_states = np.array(
            [transition.current_states for transition in replay_memory]
        ).reshape(-1, 7)
        current_qs_list = self.model.predict(current_states)

        new_states = np.array(
            [transition.new_states for transition in replay_memory]
        ).reshape(-1, 7)
        future_qs_list = self.target_model.predict(new_states)

        rewards = np.array(
            [transition.rewards for transition in replay_memory]
        ).reshape(-1, 1)
        actions = np.array(
            [transition.actions for transition in replay_memory]
        ).reshape(-1, 1)

        # Initialize X and Y arrays for training
        X = []
        Y = []

        # Combine the current states, new states, rewards, and actions into a list
        transitions = [current_states, new_states, rewards, actions]

        # Loop through each transition and calculate the new Q-value
        for index, _ in enumerate(transitions):
            max_future_q = np.max(future_qs_list[index])
            new_q = rewards[index] + DISCOUNT * max_future_q

            current_qs = current_qs_list[index]

            current_qs[actions[index]] = new_q

            X.append(current_states[index])
            Y.append(current_qs)

            # Increment the debug counter
            self.debug_counter += 1

        self.model.fit(
            np.array(X),
            np.array(Y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard],
        )

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
