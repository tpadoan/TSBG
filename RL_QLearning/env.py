import numpy as np
import networkx as nx


class ScotlandYardEnv:
    """ 
    Class representing the Scotland Yard board game as a Reinforcement Learning environment.
    """

    def __init__(self, num_detectives: int = 2):
        self.Graph = []
        self.detectives = np.array([[0]*num_detectives])