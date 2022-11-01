import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

import numpy as np
import pickle

from src.MuZero.classes import Node

def train(network_model, replay_buffer, config):

    with tf.GradientTape() as tape:

        loss = 0
        optimizer = Adam(learning_rate=config['train']['learning_rate'], beta_1=config['train']['beta_1'],
                         beta_2=config['train']['beta_2'])

        for game in replay_buffer.sample():

            game_length = len(game.reward_history)
            sampled_index = np.random.choice(
                range(game_length))  # sample an index position from the length of reward_history

            hidden_state = network_model.representation_function(game.state_history[sampled_index])

            if (sampled_index + config['train']['num_unroll_steps']) < game_length:
                num_unroll_steps = int(config['train']['num_unroll_steps'])
            else:
                num_unroll_steps = game_length - 1 - sampled_index

            for start_index in range(sampled_index, sampled_index + num_unroll_steps):
                # can only be unrolled up to the second-last time step, since every time step (start_index), we are predicting and matching values that are one time step into the future (start_index+1)

                ### get predictions ###
                hidden_state, pred_reward = network_model.dynamics_function(
                    [hidden_state, game.action_history[start_index]])
                pred_policy, pred_value = network_model.prediction_function(hidden_state)

                if (game_length - start_index - 1) >= config['train'][
                    'num_bootstrap_timesteps']:  # bootstrap using transition rewards and mcts value for final bootstrapped time step
                    true_value = sum(
                        [game.reward_history[i] * (config['self_play']['discount_factor'] ** (i - start_index)) for i in
                         range(start_index, int(start_index + config['train']['num_bootstrap_timesteps']))]) + \
                                 game.value_history[start_index + int(config['train']['num_bootstrap_timesteps'])] * \
                                 (config['self_play']['discount_factor'] ** (config['train']['num_bootstrap_timesteps']))

                else:
                    true_value = sum(
                        [game.reward_history[i] * (config['self_play']['discount_factor'] ** (i - start_index)) for i in range(start_index, game_length)])

                true_reward = game.reward_history[start_index]  # since game.reward_history is shifted, this transition reward is actually at time step (start_index+1)
                true_policy = game.policy_history[start_index + 1]  # we need to match the pred_policy at time step (start_index+1) so we need to actually index game.policy_history at (start_index+1)

                ### calculate loss ###
                loss += (1 / num_unroll_steps) * (
                        mean_squared_error(true_reward, pred_reward) + mean_squared_error(true_value, pred_value) + binary_crossentropy(true_policy, pred_policy))  # take the average loss among all unroll steps
        loss += tf.reduce_sum(network_model.representation_function.losses) + tf.reduce_sum(
            network_model.dynamics_function.losses) + tf.reduce_sum(
            network_model.prediction_function.losses)  # regularization loss

    ### update network_model weights ###
    grads = tape.gradient(loss, [network_model.representation_function.trainable_variables,
                                 network_model.dynamics_function.trainable_variables,
                                 network_model.prediction_function.trainable_variables])
    optimizer.apply_gradients(zip(grads[0], network_model.representation_function.trainable_variables))
    optimizer.apply_gradients(zip(grads[1], network_model.dynamics_function.trainable_variables))
    optimizer.apply_gradients(zip(grads[2], network_model.prediction_function.trainable_variables))

