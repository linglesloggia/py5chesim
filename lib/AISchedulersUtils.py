#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AISchedulerUtils.py contains tools for scheduling based on artificial 
intelligence algorithms.

With the use of this module it is possible to develop resource allocation 
algorithms based on artificial intelligence. It is not an algorithm in itself, 
but it provides the tools for their development. Interactions of episodes can 
be found, discriminating in which stage of learning it is. The end user will 
be the one who proposes the learning method in C{Scheds_intra.py}, making use 
of the functions of this module.
"""


import os
import pickle
from collections import deque

import numpy as np
from SchedulerUtils import *



class supervised_scheduler:
    """
    Class designed for the development of supervised schedulers.

    The end user can use this class to design supervised learning algorithms,
    e.g. Random Forest, SVM, neural networks, etc
    """

    def __init__(self, predicting_episode, scheduler):
        """
        Class constructor.

        When an object of the supervised_scheduler class is created, it is
        necessary to assign the necessary attributes to it so that it keeps
        track of the episode. Both the type of scheduler, as well as the
        feature vectors and labels required for supervised learning are
        attributes of the object.
        @param predicting_episode: episode number from which it begins to
        predict.
        @param scheduler: object of type framework_scheduler that contains the
        information about the operations to obtain the metric.
        """
        self.scheduler = scheduler
        """scheduler object type. """
        self.predicting_episode = predicting_episode
        """episode number from wich it begins to predict"""
        self.episode = 0
        """episode counter"""
        self.prediction = 0
        """predictor variable"""
        self.mlmodel = 0
        """machine learning model"""
        self.characteristics = []
        """feature array"""
        self.x_array = []
        """carries the vector of characteristics for each time"""
        self.y_array = []
        """carries the vector of labels for each time"""
        self.ind_ = 0
        """carries the value of the predicted index"""

        try:
            os.remove("data/data_vectors")
            os.remove("data/data_labels")
        except OSError as e:
            pass

        if not os.path.exists("data"):
            os.mkdir("data")

    def findInd(self, list, pred):
        """
        Auxiliary function for assignment.

        Through this function it is possible to obtain the relationship
        between the prediction and the UEs list index.
        @param list: array containing the list of UEs objects.
        @param pred: integer returned by the prediction.
        @type pred: integer.
        @return: corresponding index in the UES list, which must be assigned
        according to the prediction
        """
        if len(list) >= pred - 1:
            return pred - 1
        else:
            return 0

    def get_index(self, ues):
        """
        Main function for a supervised learning scheduler.

        This function is divided into two stages, first the model is
        learned using the scheduler of interest, running through each TTI
        as an episode, up to number C{self.predicting_episode}, then this
        model is used to predict the following assignments.The user can
        develop their supervised learning algorithms using the C{get_index}
        function, however, the user is free to choose the algorithm they want,
        for this, when the episode occurs in which C{epsiode} is equal to
        C{predicting_episode}, the user must load the model as in the example
        for the proportional fair.

        @param ues: array of ues defined in intraSliceSch.py.
        @return: the index of the UE that must be scheduled at that moment.
        """
        if self.episode <= self.predicting_episode:
            assignUEfactor(ues, calculateUEfactor(self.scheduler, self.characteristics))

            maxInd, self.ind_ = findMaxFactor(ues, self.ind_)

            aux = list(ues.keys()).index(maxInd) + 1

            if self.episode >= 40:
                with open("data/data_vectors", "ab+") as filehandle:
                    pickle.dump(np.concatenate(self.characteristics), filehandle)

                with open("data/data_labels", "ab+") as filehandle2:
                    pickle.dump(aux, filehandle2)

        if self.episode > self.predicting_episode:
            self.prediction = self.mlmodel.predict(
                [np.concatenate(self.characteristics)]
            )
            print([np.concatenate(self.characteristics)], self.prediction)

            maxInd = np.array(list(ues.keys()))[
                self.findInd(list(ues.keys()), self.prediction)
            ]
            print(list(ues.keys()))
        self.episode = self.episode + 1
        return maxInd


class qlearning_scheduler(object):
    """
    Used for the development of Q_learning schedulers.

    The user can create different schedulers based on q_learning through
    this class. A simple example of an algorithm based on Q learning can
    be found in Scheds_Intra.py.
    """

    def __init__(self, num_states, num_actions, gamma=0.95, alpha=0.5):
        self.num_states = num_states
        """number of possible states for an UE"""
        self.num_actions = num_actions
        """number of possible actions, in a common 
      scenario it could be chosen as the number of ues"""
        self.alpha = alpha
        """learning rate for q-learning algorithm."""
        self.gamma = gamma
        """discount coefficient for the q learning algorithm."""

        self.reset()

    def reset(self):
        """
        Initialize Q table with zeros.
        """
        self.Q = np.random.uniform(
            low=-2,
            high=0,
            size=([self.num_states] + [self.num_states] + [self.num_actions]),
        )  # np.zeros((self.num_states,self.num_actions))
        return

    def update(self, state, action, reward, next_state, done=0):
        """
        Update table Q for pair (state, action). if done = True, it means
        that an episode has just finished.
        @param state: variable that carries the current state, it can be
        an array of size [num_actions].
        @param action: action obtained from the EpsilonGreedyPolicy function,
        it can be the index of the user to allocate the resources.
        @param reward: reward for having performed the action.
        @param next_state: the state obtained after the action is performed.
        @param done: flag indicating if the episode ended. It is not used in
        the example version, but it can be used in a different implementation.
        """
        if done:
            td_error = reward - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
        else:
            td_error = (
                reward
                + self.gamma * np.amax(self.Q[next_state[0]][next_state[1]])
                - self.Q[state, action]
            )  # np.amax(self.Q[next_state,:], axis=0) - self.Q[state,action] 
            self.Q[state, action] += self.alpha * td_error
        return

    def EpsilonGreedyPolicy(self, eps, state):
        """
        Function that implements an epsilon-greedy policy.

        Given a Q table of dimension ([num_states]^2, num_actions),
        a state (state) and a probability of exploration (eps),
        returns the action to take: an integer between 0 and num_actions.
        With probability eps takes a random action, with probability 1-eps
        takes the action that maximizes the q table for the current state.
        When acting greedy, if there are two (or more) optimal actions,
        choose them at random.
        @param eps: value between 0 and 1, represents how likely I will
        perform a random action or follow the policy obtained.
        @param state: variable that carries the current state, it can be
        an array of size [num_actions].
        """
        if np.random.random() > eps:
            action = np.argmax(self.Q[state[0]][state[1]])
        else:
            action = np.random.randint(0, self.Q.shape[2])
        return action

class DQNAgent:
    def __init__(self, MIN_REPLAY_MEMORY_SIZE, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY, MODEL_NAME_TF, MIN_REWARD, MEMORY_FRACTION, ACTION_SPACE_SIZE, LEARNING_RATE, N_FEATURES):
        """
        Initializes a new DQNAgent object with a main and target model, a replay memory deque,
        a debug counter, and a tensorboard object.

        :return: None
        """
        self.model = self._build_model()  # Initialize the main model
        self.target_model = self._build_model()  # Initialize the target network with the same model as the main model
        self.target_model.set_weights(self.model.get_weights())  # Set the weights of the target model to be the same as the main model
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME_TF, int(time.time())))  # Create a tensorboard object for logging
        self.debug_counter = 0  # Initialize a debug counter for tracking
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # Create a replay memory deque to store the last n steps for training
        self.target_update_counter = 0  # Used to count when to update target network with main network's weights
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE  # The minimum size the replay memory must be before training
        self.MINIBATCH_SIZE = MINIBATCH_SIZE  # How many steps (samples) to use for training per batch
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY  # How often to update the target network with the main network's weights
        self.DISCOUNT = DISCOUNT  # Discount rate for calculating discounted future rewards
        self.MIN_REWARD = MIN_REWARD  # The minimum reward to consider saving a model
        self.MODEL_NAME_TF = MODEL_NAME_TF  # The name of the model
        self.MEMORY_FRACTION = MEMORY_FRACTION  # The fraction of available RAM to be used for the replay memory
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE  # The number of possible actions
        self.LEARNING_RATE = LEARNING_RATE  # The learning rate for the neural network
        self.N_FEATURES = N_FEATURES  # The number of features for the state space

    def _build_model(self):
        """
        Builds the neural network model with specified layers and activations.

        :return: A new Sequential model
        """
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=7))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE), metrics=["mae"])  # Compile the model with a mean squared error loss function and Adam optimizer
        return model

    def save_model(self, name):
        """
        Saves the model to a file with the provided name.

        :param name: The name of the file to save the model to.
        :type name: str
        :return: None
        """
        self.model.save(name)

    def load_model(self, name):
        """
        Loads the model from a file with the provided name.

        :param name: The name of the file to load the model from.
        :type name: str
        :return: None
        """
        self.model = load_model(name)

    def get_qs(self, state):
        """
        Predicts the Q-values for a given state.

        :param state: The state for which to predict Q-values.
        :type state: np.array
        :return: A np.array of predicted Q-values.
        """
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, replay_memory):
        """
        Trains the neural network on the provided replay memory.

        :param replay_memory: A deque of previous state, action, reward, and next state tuples.
        :type replay_memory: collections.deque
        :return: None
        """
        if len(replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:  # If there is not enough data in the replay memory, do not train
            return

        # Get the current states, new states, rewards, and actions from the replay memory
        current_states = np.array([transition.current_states for transition in replay_memory]).reshape(-1, self.N_FEATURES)
        current_qs_list = self.model.predict(current_states)

        new_states = np.array([transition.new_states for transition in replay_memory]).reshape(-1, self.N_FEATURES)
        future_qs_list = self.target_model.predict(new_states)

        # Convert the rewards and actions in the replay memory to numpy arrays
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
            new_q = rewards[index] + self.DISCOUNT * max_future_q

            current_qs = current_qs_list[index]

            current_qs[actions[index]] = new_q

            X.append(current_states[index])
            Y.append(current_qs)

            # Increment the debug counter
            self.debug_counter += 1

        # Fit the model to the training data
        self.model.fit(
            np.array(X),
            np.array(Y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False
        )

        # Update the target network weights every UPDATE_TARGET_EVERY steps
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        