import networkx as nx
import numpy as np

import utils.graph_util

def valid_moves_list(graph: nx.Graph, mrX: np.ndarray[int]) -> np.ndarray[int]:
    """ Get the list of valid moves that can be done by mrX.

    Args:
        graph (nx.Graph): The nodes' game graph.
        mrX (np.ndarray): MrX information array.

    Returns:
        np.ndarray[int]: The list of valid moves.
    """
    # Get the current node in which mrX is
    current_node = mrX[0]
    # Get the encoding of all the edges that can be reached from the current node
    edges = utils.graph_util.connections(graph, current_node)

    return np.array(edges)