import torch.nn as nn
import numpy as np

import env

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
        length (List[float]): Distance at the end of the game of each detective w.r.t mrX
    """

    def __init__(self, model_mrX: nn.Module, model_detectives: np.ndarray(nn.Module), explore: float = 0.0):
        # Environment 
        self.env = env.ScotlandYardEnv()
        self.explore = explore
        self.reward = 0

        # Models
        self.model_mrX = model_mrX
        self.model_detectives = model_detectives 
        
        # Observations and predictions for each model
        self.mrX_obs = []
        self.mrX_y = []
        self.detective_obs = {i: [] for i in range(len(model_detectives))}
        self.detective_y = {i: [] for i in range(len(model_detectives))} 
        
        # Metadata
        self.length = []
