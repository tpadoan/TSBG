import torch.nn as nn
import numpy as np
import RL_QLearning.env as env
from models.detective import DetectiveModel
from models.mrX import MrXModel
import utils.graph_util

class QLearning:
    """ Class containing the methods and execution steps of the QLearning algorithm.

    Attributes:
        env (env.ScotlandYardEnv): The Scotland Yard RL environment.
        explore (float): The exploration term, it decreases with the increase of the number of epochs spent in the training phase.
        model_mrX (nn.Module): MrX neural network model.
        model_detectives (np.ndarray(nn.Module)): Detectives nerual network models.
        mrX_obs (List[List[int]]): List of states observed by mrX.
        mrX_y (List[List[float]]): List of q_values used by mrX.
        detective_obs (Dict[int, List[int]]): Dictionary of states observed by each detective.
        detective_y (Dict[int, List[List[float]]]): Dictionary of q_values used by each detective.
        length (List[float]): Distance at the end of the game of each detective w.r.t mrX.
    """

    def __init__(self, model_mrX: MrXModel, model_detectives: np.ndarray[DetectiveModel], max_turns: int = 10, explore: float = 0.0, interact: bool = False):
        # Environment 
        self.env = env.ScotlandYardEnv(num_detectives=len(model_detectives), num_max_turns=max_turns, interactive=interact)
        self.explore = explore
        self.reward = 0

        # Models
        self.model_mrX = model_mrX
        self.model_detectives = model_detectives 
        
        # Observations and predictions for each model
        self.mrX_obs = []
        self.mrX_y = []
        self.detective_obs = {i: [] for i in range(1, len(model_detectives)+1)}
        self.detective_y = {i: [] for i in range(1, len(model_detectives)+1)} 
        
        # Metadata
        self.length = []

    def run_episode(self):
        self.env.initialize_game()
        done = False

        while not done:
            current_observation, sub_turn_counter = self.env.observe()
            actions = self.env.get_valid_moves()
            action_probs = np.full(actions.shape[0], self.explore / actions.shape[0], dtype=float)
            # Switch allows to use an heuristic to decide the next position for mrX
            # switch = False
            if sub_turn_counter != 0:
                best_action, _ = self.get_best_action(current_observation, actions, self.model_detectives[sub_turn_counter-1])
                action_probs[best_action] += (1. - self.explore)
            else:
                # Probabilistic, based on squared min distance from detectives
                dists = [0]*actions.shape[0]
                for i in range(actions.shape[0]):
                    dists[i] = self.env.min_shortest_path(actions[i][1])**2
                tot = sum(dists)
                if not tot:
                    action_probs[0] += (1 - self.explore)
                else:
                    for i in range(len(action_probs)):
                        action_probs[i] = dists[i] / tot
                # Best action selection
                # best_action, _ = self.get_best_action(current_observation, actions, self.model_mrX)
                # # Random heuristic
                # rand = random.randint(0, actions.shape[0]-1)
                # next_node = actions[rand][1]
                # mrX heuristic
                # min_node, max_node = float('inf'), -1
                # for action in actions:
                #     action_node = action[1]
                #     if switch:
                #         if action_node > max_node:
                #             next_node, max_node, transport_encoding = action_node, action_node, action[2:]
                #     else:
                #         if action_node < min_node:
                #             next_node, min_node, transport_encoding = action_node, action_node, action[2:]
                
                # switch = not switch  # Toggle the switch (0 to 1 or 1 to 0)
                # next_observation, reward, done = self.env.take_action(next_node, transport_encoding)
                # continue

            action_taken = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_node = actions[action_taken][1]
            transport_encoding = actions[action_taken][2:]
            state_used = current_observation.tolist() + utils.graph_util.node_one_hot_encoding(next_node, self.env.G.number_of_nodes()) + transport_encoding.tolist()
            next_observation, reward, done = self.env.take_action(next_node, transport_encoding)
            actions = self.env.end_turn_valid_moves()

            # If there are no available actions then we stay in the node
            if actions.shape[0] == 0:
                actions = np.array([[next_node, next_node, 0, 0, 0]])

            # Update the observation and Q value lists based on the player
            if sub_turn_counter:
                _, Q_max = self.get_best_action(next_observation, actions, self.model_detectives[sub_turn_counter-1])
                self.detective_obs[sub_turn_counter].append(state_used)
                self.detective_y[sub_turn_counter].append([Q_max])
            # else:
                # _, Q_max = self.get_best_action(next_observation, actions, self.model_mrX)
                # self.mrX_obs.append(state_used)
                # self.mrX_y.append([Q_max])
        
        # Since the game is over, let's compute the distances of each detective w.r.t. mrX
        for i in range(len(self.model_detectives)):
            self.length.append(self.env.shortest_path(i))

        self.reward = reward
        self.q_learn()

        return self.reward, self.model_mrX, self.model_detectives

    def q_learn(self):
        """ Learning phase for the detectives.
        """
        gamma = 0.9
        for i, detective_obs in self.detective_obs.items():
            num_of_observations = len(detective_obs)
            if num_of_observations != 0:
                reward = self.reward
                for j in range(num_of_observations):
                    multiplier = num_of_observations - j
                    self.detective_y[i][j][0] += gamma * reward  * multiplier
                self.model_detectives[i-1].optimize(self.detective_obs[i], self.detective_y[i])

    def get_best_action(self, current_state: np.ndarray[int], actions: np.ndarray[int], model: nn.Module) -> tuple[int, float]:
        """ QLearning update rule implementation.

        Args:
            current_state (np.ndarray[int]): The current state.
            actions (np.ndarray[int]): The possible actions that can be done.
            model (nn.Module): The neural network model to be used.

        Returns:
            tuple[int, float]: The best action to take and the corresponding Q value.
        """
        observation = [[] for _ in range(actions.shape[0])]
        for i in range(actions.shape[0]):
            next_node = utils.graph_util.node_one_hot_encoding(actions[i][1], num_nodes= self.env.G.number_of_nodes())
            observation[i] = current_state.tolist() + next_node + actions[i][2:].tolist()
        Q_values = model.predict(observation)

        return np.argmax(Q_values), np.amax(Q_values)