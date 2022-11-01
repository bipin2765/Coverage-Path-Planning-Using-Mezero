from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras.layers import Input,Dense,Concatenate

import numpy as np
np.random.bit_generator = np.random._bit_generator

import os


class Node:
    """
    A class that represents the nodes used in Monte Carlo Tree Search.
    """

    def __init__(self, prior):

        self.prior = prior  # prior probability given by the output of the prediction function of the parent node

        self.hidden_state = None  # from dynamics function
        self.transition_reward = 0  # from dynamics function
        self.policy = None  # from prediction function
        self.value = None  # from prediction function

        self.is_expanded = False
        self.children = []

        self.cumulative_value = 0  # divide this by self.num_visits to get the mean Q-value of this node
        self.num_visits = 0

    def expand_node(self, parent_hidden_state, parent_action, network_model):

        # get hidden state representation and transition reward
        hidden_state, transition_reward = network_model.dynamics_function([parent_hidden_state, parent_action])
        self.hidden_state = hidden_state
        self.transition_reward = transition_reward.numpy()[0][0]  # convert to scalar

        # get predicted policy and value
        policy, value = network_model.prediction_function(self.hidden_state)
        self.policy = policy
        self.value = value.numpy()[0][0]  # convert to scalar

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append(Node(self.policy.numpy()[0][action]))

        self.is_expanded = True

    def expand_root_node(self, current_state, network_model):
        # get hidden state representation
        hidden_state = network_model.representation_function(current_state.reshape(1, -1))
        self.hidden_state = hidden_state
        self.transition_reward = 0  # no transition reward for the root node

        # get predicted policy and value
        policy, value = network_model.prediction_function(self.hidden_state)
        self.policy = policy
        self.value = value.numpy()[0][0]  # convert to scalar

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append(Node(self.policy.numpy()[0][action]))

        self.is_expanded = True

    def get_ucb_score(self, visit_sum, min_q_value, max_q_value, config):

        normalized_q_value = self.transition_reward + config['self_play'][
            'discount_factor'] * self.cumulative_value / max(self.num_visits, 1)
        if min_q_value != max_q_value: normalized_q_value = (normalized_q_value - min_q_value) / (
                    max_q_value - min_q_value)  # min-max normalize q-value, to make sure q-value is in the interval [0,1]
        # if min and max value are equal, we would end up dividing by 0

        return normalized_q_value + \
               self.prior * np.sqrt(visit_sum) / (1 + self.num_visits) * \
               (config['mcts']['c1'] + np.log((visit_sum + config['mcts']['c2'] + 1) / config['mcts']['c2']))
