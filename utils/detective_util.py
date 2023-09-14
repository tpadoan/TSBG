import networkx as nx
import numpy as np

import utils.graph_util

def valid_moves_list(graph: nx.Graph, detectives: np.ndarray[int], detective_id: int) -> np.ndarray[int]:
    """ Get the list of valid moves that can be done by detective i.

    Args:
        graph (nx.Graph): The nodes' game graph.
        detectives (np.ndarray): The array of detectives.
        detective_id (int): The current detective id.

    Returns:
        np.ndarray[int]: The list of valid moves.
    """
    # Get the current node in which the detective is
    current_node = detectives[detective_id][0]
    # Get the encoding of all the edges that can be reached from the current node
    edges = utils.graph_util.connections(graph, current_node)
    # Filter the possible destinations by avoiding to land in a node where another detective is
    valid_list = [edge for edge in edges if all(detective[0] != edge[1] for detective in detectives)]

    return np.array(valid_list)