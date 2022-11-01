import copy
import tqdm
import distutils.util

from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay
from src.base.GridActions import GridActions
from src.CPP.State import CPPState
from src.MuZero.Agent_ import MuZeroAgent
from src.MuZero.classes import *
from src.MuZero.replay_memory import ReplayBuffer
from src.MuZero.Train import train

class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()

confignew = {  'action_size' : 5,
               'mcts': { 'num_simulations': 1e3, # number of simulations to conduct, every time we call MCTS
                         'c1': 1.25, # for regulating MCTS search exploration (higher value = more emphasis on prior value and visit count)
                         'c2': 19625 }, # for regulating MCTS search exploration (higher value = lower emphasis on prior value and visit count)
               'self_play': { 'num_games': 5000, # number of games the agent plays to train on
                              'discount_factor': 1.0, # used when backpropagating values up mcts, and when calculating bootstrapped value during training
                              'save_interval': 100}, # how often to save network_model weights and replay_buffer
               'replay_buffer': { 'buffer_size': 1e3, # size of the buffer
                                  'sample_size': 1e2}, # how many games we sample from the buffer when training the agent
               'train': { 'num_bootstrap_timesteps': 500, # number of timesteps in the future to bootstrap true value
                          'num_unroll_steps': 1e1, # number of timesteps to unroll to match action trajectories for each game sample
                          'learning_rate': 1e-3, # learning rate for Adam optimizer
                          'beta_1': 0.9, # parameter for Adam optimizer
                          'beta_2': 0.999}, # parameter for Adam optimizer
               'test': { 'num_test_games': 10, # number of times to test the agent using greedy actions
                         'record': True},
               'seed': 0
               }

class BufferMemory:
    def __int__(self, init_state = None):
        if init_state:
            self.state_history = [init_state]  # starts at t = 0
        self.action_history = []  # starts at t = 0
        self.reward_history = []  # starts at t = 1 (the transition reward for reaching state 1)
        self.value_history = []  # starts at t = 0 (from MCTS)
        self.policy_history = []  # starts at t = 0 (from MCTS)

class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.agent: MuZeroAgent = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 1
        self.step_count = 0
        self.action_size = 5

    def train_episode(self):
        # Create game memory object to store play history
        _init_state = self.agent.state_size.reshape(1, -1) #states including boolean and scaler in single row.
        memory_: BufferMemory = BufferMemory(_init_state)

        #self.env.seed(int(np.random.choice(range(int(1e5)))))
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step(state, memory_)
        self.replay_buffer.add(memory_)
        train(self.agent.Network_model, self.replay_buffer, confignew)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)


        self.episode_count += 1


    def run(self):

        self.replay_buffer = ReplayBuffer(confignew)
        print(type(self.agent))
        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state, memory_: BufferMemory):
        if type(state) == CPPState:
            state = self.agent.Network_model.transfrom_state(state)

        old_state = copy.deepcopy(state)
        action_index = mcts(old_state, self.agent.Network_model, get_temperature(self.episode_count),confignew)

        current_state = self.physics.step(GridActions(action_index))
        reward = self.rewards.calculate_reward(old_state, GridActions(action_index), current_state)
        action = np.array([1 if i==action_index else 0 for i in range(self.action_size)]).reshape(1,-1)

        memory_.state_history.append(current_state.reshape(1, -1))
        memory_.action_history.append(action)
        memory_.reward_history.append(reward)
        #self.trainer.add_experience(current_state, action, reward)
        #self.stats.add_experience((current_state, action, reward)
        self.step_count += 1

        return copy.deepcopy(current_state)


    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass


def mcts(reset_state, network_model, temperature, confignew):


    root_node = Node(0)
    root_node.expand_root_node(reset_state, network_model)
    game_memory: BufferMemory = BufferMemory()

    min_q_value, max_q_value = root_node.value, root_node.value  # keep track of min and max mean-Q values to normalize them during selection phase
    # this is for environments that have unbounded Q-values, otherwise the prior could potentially have very little influence over selection, if Q-values are large
    for _ in range(int(confignew['mcts']['num_simulations'])):
        current_node = root_node

        # SELECT a leaf node
        search_path = [root_node]  # node0, ... (includes the final leaf node)
        action_history = []  # action0, ...
        while current_node.is_expanded:
            # total_num_visits need to be at least 1
            # otherwise when selecting for child nodes that haven't been visited, their priors won't be taken into account, because it'll be multiplied by total_num_visits in the UCB score, which is zero
            total_num_visits = max(1, sum([current_node.children[i].num_visits for i in
                                           range(len(current_node.children))]))

            action_index = np.argmax(
                [current_node.children[i].get_ucb_score(total_num_visits, min_q_value, max_q_value, confignew) for i in
                 range(len(current_node.children))])
            current_node = current_node.children[action_index]

            search_path.append(current_node)
            action_history.append(
                np.array([1 if i == action_index else 0 for i in range(confignew['action_size'])]).reshape(1, -1))

        # EXPAND selected leaf node
        current_node.expand_node(search_path[-2].hidden_state, action_history[-1], network_model)

        # BACKPROPAGATE the bootstrapped value (approximated by the network_model.prediction_function) to all nodes in the search_path
        value = current_node.value
        for node in reversed(search_path):
            node.cumulative_value += value
            node.num_visits += 1

            node_q_value = node.cumulative_value / node.num_visits
            min_q_value, max_q_value = min(min_q_value, node_q_value), max(max_q_value,
                                                                           node_q_value)  # update min and max values

            value = node.transition_reward + confignew['self_play'][
                'discount_factor'] * value  # updated for parent node in next iteration of the loop

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([root_node.children[i].num_visits for i in range(len(root_node.children))])
    policy = np.array([root_node.children[i].num_visits / total_num_visits for i in range(len(root_node.children))])

    if temperature == None:  # take the greedy action (to be used during test time)
        action_index = np.argmax(policy)
    else:  # otherwise sample (to be used during training)
        policy = (policy ** (1 / temperature)) / (policy ** (1 / temperature)).sum()
        action_index = np.random.choice(range(network_model.action_size), p=policy)

    # update Game search statistics
    game_memory.value_history.append(
        root_node.cumulative_value / root_node.num_visits)  # use the root node's MCTS value as the ground truth value when training
    game_memory.policy_history.append(policy)  # use the MCTS policy as the ground truth value when training

    return action_index


def get_temperature(num_iter):
    """
    This function regulates exploration vs exploitation when selecting actions during self-play.
    Given the current interation number of the learning algorithm, return the temperature value to be used by MCTS.

    Args:
        num_iter (int): The number of iterations that have passed for the learning algorithm

    Returns:
        temperature (float): Controls the level of exploration of MCTS (the lower the number, the greedier the action selection)
    """

    # as num_iter increases, temperature decreases, and actions become greedier
    if num_iter < 100:
        return 3
    elif num_iter < 200:
        return 2
    elif num_iter < 300:
        return 1
    elif num_iter < 400:
        return .5
    elif num_iter < 500:
        return .25
    elif num_iter < 600:
        return .125
    else:
        return .0625



