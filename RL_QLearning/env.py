import numpy as np
import networkx as nx

import utils.graph_util


class ScotlandYardEnv:
    """ 
    Class representing the Scotland Yard board game as a Reinforcement Learning environment.

    Attributes:
        G (nx.Graph): Board nodes graph.
        detectives (np.ndarray): Array containing current position of each detective involved in mrX's chase.
        mrX (np.ndarray): mrX current position.
        mrX_locations (np.ndarray): Array of locations that mrX will visit during the game (at most == maximum number of turns played).
        starting_nodes (np.ndarray): Starting node of each player.
        turn_number (int): Game turn number.
        turn_sub_counter (int): Player turn inside a single game turn. 0 means it's mrX's turn, otherwise it's i-th detective's turn.
        completed (bool): Boolean to keep track of the game status.
        reward (float): Reward assigned to the detectives.
        mrX_can_move (bool): Boolean to tell whether mrX can move or not.
        detective_can_move (List[bool]): Boolean to tell whether each detective can move or not.
    """
    def __init__(self, num_detectives: int = 2, num_max_turns: int = 10):
        self.G = []
        self.detectives = np.array([[0]*num_detectives])
        self.mrX = np.array([])
        self.mrX_locations = np.array([0]*num_max_turns)
        self.starting_nodes = None
        self.turn_number = 0
        self.turn_sub_counter = 0
        self.completed = False
        self.reward = 0
        self.mrX_can_move = True
        self.detective_can_move = [True] * num_detectives

        ### Metadata utilites
        self.detectives_location_log = np.array([self.mrX_locations] * 5)
        self.last_move_by_which_player = 0

    def initialize_game(self):
        # Initialize the game graph
        self.G = utils.graph_util.generate_graph()
        # Initialize the starting nodes
        num_players = 1 + len(self.detectives)
        self.starting_nodes = np.random.choice(np.array(range(1,26)), size=num_players, replace=False)
        # Initialize mrX starting node
        self.mrX = self.starting_nodes[0]
        # Initialize the detectives' starting nodes
        for i in range(len(self.detectives)):
            self.detectives[i] = self.starting_nodes[i+1]

