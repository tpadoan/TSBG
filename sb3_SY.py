import stable_baselines3
import gym
from gym.spaces import Discrete, MultiDiscrete, Dict, Box
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils.graph_util, utils.detective_util, utils.mrX_util

class ScotlandYard(gym.Env):
    def __init__(self, steps : int, G: nx.Graph, starting_nodes, num_detectives):
        self.G = G
        self.steps = steps
        self.starting_nodes = starting_nodes
        self.num_detectives = num_detectives
        # Action space
        self.action_space = Discrete(len(self.G.number_of_nodes())**2)
        # State space description
        # The state will be the one hot encoding of mrX position followed by the i-th detective position
        self.observation_space = MultiDiscrete([1] * 3*len(self.G.number_of_nodes()))

        self.state = [utils.graph_util.node_one_hot_encoding(node) for node in starting_nodes]

        self.turn_number = 0
        self.turn_sub_counter = 0

        self.detectives = np.array([[0] for _ in range(num_detectives)])
        self.mrX = np.array([self.starting_nodes[0]])
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i+1]

        self.interactive = False
        self.done = False
        self.reward = 0
        self.img = plt.imread('data/graph.png')

    def step(self, action, evaluation = False):
        # Decrease the steps for RL agent
        self.steps -= 1
        valid_moves = self.get_valid_moves()
        valid_dst = [move[1] for move in valid_moves]
        # When the action leads to an invalid node, ignore the iteration
        if action not in valid_dst:
            self.reward = 0
            self.done = False
            return self.state, self.reward, self.done
        
        else:
            next_node = action
            # Let the corresponding player play
            if self.turn_sub_counter == 0:
                self.mrX[0] = next_node
            else:
                self.detectives[self.turn_sub_counter-1] = next_node

            self.game_step()

            # Take the observation of the new environment
            self.state, _ = self.observe()

            self.turn_sub_counter += 1

            # One game turn is over
            if self.turn_sub_counter > self.num_detectives:
                self.turn_sub_counter = 0
                self.turn_number += 1

            return self.state, self.reward, self.done
        
    def game_step(self):
        """ Make a step in the environment and check whether the game is finished or not
        """
        if self.interactive:
            self.render()
            if input(f"sub_turn {self.turn_sub_counter} done, continue?\t") == 'n':
                quit()

        # Check that mrX can still move
        if utils.mrX_util.valid_moves_list(self.G, self.mrX).size == 0:
            self.mrX_can_move = False
        else:
            self.mrX_can_move = True

        # Check that any detectives can still move
        self.detective_can_move = [utils.detective_util.valid_moves_list(self.G, self.detectives, i).size != 0 for i in range(len(self.detectives))]
        all_detectives_cant_move = not any(self.detective_can_move)

        # Add local reward for each detective
        if self.turn_sub_counter != 0:
            self.reward = 1-self.shortest_path(self.turn_sub_counter-1)/self.max_min_distance

        # If that is not the case, then the game is over
        if all_detectives_cant_move:
            self.done = True
            self.reward = -1

        # Check that mrX is not in the same position as one of the detectives
        for detective in self.detectives:
            if detective[0] == self.mrX[0]:
                self.done = True
                self.reward = 1
                return

        # The last scenario is the one in which we reached the maximum number of turns
        if self.turn_number == self.max_turns-1 and self.turn_sub_counter == self.num_detectives:
            self.done = True
            self.reward = -1


    def get_valid_moves(self) -> np.ndarray[int]:
        """ Array of valid moves to play.

        Returns:
            np.ndarray[int]: The valid moves array.
        """
        if self.turn_sub_counter != 0:
            return utils.detective_util.valid_moves_list(self.G, self.detectives, self.turn_sub_counter-1)
        else:
            return utils.mrX_util.valid_moves_list(self.G, self.mrX)

    def observe(self) -> tuple[np.ndarray[int], int]:
        """ Observe the environment based on which player is playing.

        Returns:
            tuple[np.ndarray[int], int]: Current state and current turn sub counter.
        """
        # If it is one of the detectives' turn
        if self.turn_sub_counter != 0:
            return (self.observe_as_detective(), self.turn_sub_counter)
        # Otherwise, it is mrX turn
        else:
            return (None, self.turn_sub_counter) # (self.observe_as_mrX(), self.turn_sub_counter)

    def observe_as_mrX(self) -> np.ndarray[int]:
        """ The observation from mrX point of view is made of its position, the detectives ones and the turn number.

        Returns:
            np.ndarray[int]: The current observation of mrX.
        """
        # Add mrX current position
        observation = utils.graph_util.node_one_hot_encoding(node_id = self.mrX[0], num_nodes = self.G.number_of_nodes())
        # Add each detective current position
        for i in range(len(self.detectives)):
            observation.extend(utils.graph_util.node_one_hot_encoding(node_id = self.detectives[i][0], num_nodes = self.G.number_of_nodes()))

        return np.array(observation)

    def observe_as_detective(self) -> np.ndarray[int]:
        """ The observation from detectives point of view.

        Returns:
            np.ndarray[int]: The current observation of the detectives.
        """
       # Add mrX current position
        observation = utils.graph_util.node_one_hot_encoding(node_id = self.mrX[0], num_nodes = self.G.number_of_nodes())
        # Add each detective current position
        for i in range(len(self.detectives)):
            observation.extend(utils.graph_util.node_one_hot_encoding(node_id = self.detectives[i][0], num_nodes = self.G.number_of_nodes()))

        return np.array(observation)

    def render(self):
        plt.clf()
        plt.imshow(self.img)
        plt.axis('off')
        X = []
        Y = []
        for v in self.state[2]:
            x,y = utils.graph_util.get_coords(self.G, v)
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, 'o', ms=11, color='none', mec='magenta')
        X = []
        Y = []
        for v in self.state[0]:
            x,y = utils.graph_util.get_coords(self.G, v)
            X.append(x)
            Y.append(y)
        plt.plot(X, Y, 'D', ms=9, color='none', mec='cyan')
        if self.state[1]:
            x,y = utils.graph_util.get_coords(self.G, self.state[1])
            plt.plot(x, y, '*', ms=10, color='none', mec='gold')
        plt.show()

    def reset(self):
        self.detectives = np.array([[0] for _ in range(len(self.detectives))])
        self.mrX = np.array([self.starting_nodes[0]])
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i+1]
        self.done = False
        self.reward = 0
        self.state = [utils.graph_util.node_one_hot_encoding(node) for node in self.starting_nodes]

        return self.state

if __name__ == "__main__":
    pass