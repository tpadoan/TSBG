import numpy as np
import networkx as nx
import utils.graph_util, utils.detective_util, utils.mrX_util
import matplotlib.pyplot as plt

class ScotlandYardEnv:
    """ 
    Class representing the Scotland Yard board game as a Reinforcement Learning environment.

    Attributes:
        G (nx.Graph): Board nodes graph.
        detectives (np.ndarray): Array containing current position of each detective involved in mrX's chase.
        mrX (np.ndarray): mrX current position.
        det_locations (list): List of lists of locations that each detective will visit during the game (at most == number of detective times maximum number of turns played).
        mrX_locations (list): List of locations that mrX will visit during the game (at most == maximum number of turns played).
        mrX_transport_log (np.ndarray): Array of transports used by mrX in each turn
        starting_nodes (np.ndarray): Starting node of each player.
        turn_number (int): Game turn number.
        turn_sub_counter (int): Player turn inside a single game turn. 0 means it's mrX's turn, otherwise it's i-th detective's turn.
        reveals (list): List of the turns when mr.X must reveal its current position (after moving)
        completed (bool): Boolean to keep track of the game status.
        reward (float): Reward assigned to the detectives.
        mrX_can_move (bool): Boolean to tell whether mrX can move or not.
        detective_can_move (List[bool]): Boolean to tell whether each detective can move or not.
        random_start (bool): Flag stating wether initial positions of players should be chosen randomly or not.
        interactive (bool): Flag stating wether the program should await some input before continuing executing after each move.
    """
    def __init__(self, num_detectives: int = 2, num_max_turns: int = 10, random_start: bool = True, interactive: bool = False):
        self.G = None
        self.detectives = np.array([[0] for _ in range(num_detectives)])
        self.mrX = np.array([])
        self.state = [[0 for _ in range(num_detectives)], 0, {0}]
        self.det_locations = [[0]*num_max_turns for i in range(num_detectives)]
        self.mrX_locations = [0]*num_max_turns
        self.mrX_transport_log = np.array([[0,0,0]]*num_max_turns)
        self.starting_nodes = None
        self.turn_number = 0
        self.turn_sub_counter = 0
        self.reveals = [i for i in range(1, num_max_turns+1)]
        self.completed = False
        self.reward = 0
        self.mrX_can_move = True
        self.detective_can_move = [True] * num_detectives
        self.random_start = random_start
        self.interactive = interactive

        ### Metadata utilites
        self.last_move_by_which_player = 0
        self.num_detectives = num_detectives
        self.max_turns = num_max_turns

        self.img = plt.imread('data/graph.png')
        plt.ion()

    def initialize_game(self):
        """ Initialize the game by generating the game graph and choosing (randomly) the starting nodes for each player.
        """
        # Initialize the game graph
        self.G = utils.graph_util.generate_graph()
        # Initialize the starting nodes
        if self.random_start or self.num_detectives > 3:
            self.starting_nodes = np.random.choice(np.array(range(1,self.G.number_of_nodes()+1)), size=1+self.num_detectives, replace=False)
        else:
            self.starting_nodes = [5] + [20-7*i for i in range(self.num_detectives)]
        self.max_min_distance = max([values for key in dict(nx.all_pairs_shortest_path_length(self.G)) for values in dict(nx.all_pairs_shortest_path_length(self.G))[key]])
        # Initialize mrX starting node
        self.mrX = np.array([self.starting_nodes[0]])
        self.state[1] = self.mrX[0]
        # self.state[2] = {self.mrX[0]:1.0}
        self.state[2] = {self.mrX[0]}
        # Initialize the detectives' starting nodes
        for i in range(self.num_detectives):
            self.detectives[i] = self.starting_nodes[i+1]
            self.state[0][i] = self.starting_nodes[i+1]
        if self.interactive:
            self.drawMap()
            if input("Begin?\t") == 'n':
                quit()

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

        observation += [self.turn_number]

        return np.array(observation)

    def observe_as_detective(self) -> np.ndarray[int]:
        """ The observation from detectives point of view.

        Returns:
            np.ndarray[int]: The current observation of the detectives.
        """
        # Possible positions of mr.X taking into account the moves and the starting position
        observation = utils.graph_util.nodes_ohe(self.state[2], self.G.number_of_nodes())
        observation.extend(utils.graph_util.node_one_hot_encoding(self.detectives[self.turn_sub_counter-1][0], self.G.number_of_nodes()))
        return np.array(observation) ##############################################################
        # Inform which detective is playing
        observation += [1 if i == self.turn_sub_counter-1 else 0 for i in range(len(self.detectives))]
        # Add each detective current position
        for i in range(len(self.detectives)):
            observation.extend(utils.graph_util.node_one_hot_encoding(self.detectives[i][0], self.G.number_of_nodes()))
        # Add the log of the transports used by mrX in the whole game
        for transport_log in self.mrX_transport_log:
            observation.extend(transport_log.tolist())
        return np.array(observation)

    def propagate(self, move: str):
        new = set()
        for s in self.state[2]:
            new = new.union(utils.graph_util.destinations_by(self.G, s, move))
        self.state[2] = new.difference(self.state[0])

    def propagate_prob(self, move: str):
        new = {}
        tot = 0
        for node,prob in self.state[2].items():
            succ = utils.graph_util.destinations_by(self.G, node, move).difference(self.state[0])
            size = len(succ)
            for s in succ:
                p = prob/size
                if s in new:
                    new[s] += p
                else:
                    new[s] = p
                tot += p
        for node,prob in new.items():
            new[node] = prob/tot
        self.state[2] = new

    def get_valid_moves(self) -> np.ndarray[int]:
        """ Array of valid moves to play.

        Returns:
            np.ndarray[int]: The valid moves array.
        """
        if self.turn_sub_counter != 0:
            return utils.detective_util.valid_moves_list(self.G, self.detectives, self.turn_sub_counter-1)
        else:
            return utils.mrX_util.valid_moves_list(self.G, self.mrX)

    def end_turn_valid_moves(self) -> np.ndarray[int]:
        """ Array of valid moves to play by the end of the turn.

        Returns:
            np.ndarray[int]: The valid moves array.
        """
        if self.last_move_by_which_player == 0:
            return utils.mrX_util.valid_moves_list(self.G, self.mrX)
        else:
            return utils.detective_util.valid_moves_list(self.G, self.detectives, self.last_move_by_which_player-1)

    def take_action(self, next_node: int, transport_one_hot: np.ndarray[int]):
        if self.completed:
            print("Game over!")
            return
        if self.turn_sub_counter == 0:
            self.play_mrX(next_node, transport_one_hot)
        else:
            self.play_detective(next_node)
        # Make a step in the environment
        self.step()
        # Take the observation of the new environment
        observation, _ = self.observe()
        # Make the next player play
        self.last_move_by_which_player = self.turn_sub_counter
        self.turn_sub_counter += 1
        # One game turn is over
        if self.turn_sub_counter > self.num_detectives:
            self.turn_sub_counter = 0
            self.turn_number += 1

        # In the case one of the player cannot move anymore, we let the other ones play
        self.skip_turn()

        return observation, self.reward, self.completed

    def play_mrX(self, next_node: int, transport_one_hot: np.ndarray[int]):
        """ Make mrX play.

        Args:
            next_node (int): New reached node.
            transport_one_hot (np.ndarray[int]): One-hot encoding of the transport used by mrX to reach the new node.
        """
        # Update mrX position and state
        self.mrX[0] = next_node
        self.state[1] = next_node
        if self.turn_number+1 in self.reveals:
            # self.state[2] = {next_node:1.0}
            self.state[2] = {next_node}
        else:
            if transport_one_hot[0]:
                self.propagate('boat')
            elif transport_one_hot[1]:
                self.propagate('tram')
            else:
                self.propagate('cart')
        # Register the new position in the locations log
        self.mrX_locations[self.turn_number] = next_node
        # Register the used transport in the transport log
        self.mrX_transport_log[self.turn_number] = transport_one_hot
    
    def play_detective(self, next_node: int):
        """ Make a detective play.

        Args:
            next_node (int): New reached node.
            transport_one_hot (np.ndarray[int]): One-hot encoding of the transport used by the detective to reach the new node.
        """
        # Simply update the position of the detective
        self.detectives[self.turn_sub_counter-1] = next_node
        self.state[0][self.turn_sub_counter-1] = next_node
        # diff = self.state[2].pop(next_node, False):
        # if diff:
        #    tot = 1.0 - diff
        #    for node,prob in self.state[2].items():
        #        self.state[2][node] = prob/tot
        self.state[2].discard(next_node)
        self.det_locations[self.turn_sub_counter-1][self.turn_number] = next_node

    def step(self):
        """ Make a step in the environment and check whether the game is finished or not
        """
        if self.interactive:
            self.drawMap()
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
            self.completed = True
            self.reward = -1

        # Check that mrX is not in the same position as one of the detectives
        for detective in self.detectives:
            if detective[0] == self.mrX[0]:
                self.completed = True
                self.reward = 1
                return

        # The last scenario is the one in which we reached the maximum number of turns
        if self.turn_number == self.max_turns-1 and self.turn_sub_counter == self.num_detectives:
            self.completed = True
            self.reward = -1

    def skip_turn(self):
        """ Function to let the other players play in the case one cannot move anymore
        """
        if self.turn_sub_counter == 0 and not self.mrX_can_move:
            self.turn_sub_counter += 1
            self.step()
            if not self.completed:
                self.skip_turn()

        elif self.turn_sub_counter > 0 and not self.detective_can_move[self.turn_sub_counter - 1]:
            self.turn_sub_counter += 1
            if self.turn_sub_counter > self.num_detectives:
                self.turn_sub_counter = 0
                self.turn_sub_counter += 1
            self.step()
            if not self.completed:
                self.skip_turn()

    def shortest_path(self, det_id):
        return nx.shortest_path_length(self.G, self.detectives[det_id][0], self.mrX[0])

    def min_shortest_path(self, node):
        md = nx.shortest_path_length(self.G, node, self.detectives[0][0])
        for i in range(1, len(self.detectives)):
            d = nx.shortest_path_length(self.G, node, self.detectives[i][0])
            if d < md:
                md = d
        return md

    def drawMap(self):
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