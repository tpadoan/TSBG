import numpy as np
import networkx as nx

def generate_graph() -> nx.Graph:
    """Generate the game graph from the adjacency lists described in each transport.txt file

    Returns:
        nx.Graph: The game graph.
    """
    # Specify each node coordinate
    G = nx.MultiGraph()
    with open('data(coords).txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id, x_coord, y_coord = map(int, parts)
            G.add_node(node_id, pos=(x_coord, y_coord))

    # Create graphs corresponding to each transport vehicle
    boat = nx.read_adjlist("data/boat.txt", create_using=nx.MultiGraph(), nodetype=int)
    cart = nx.read_adjlist("data/cart.txt", create_using=nx.MultiGraph(), nodetype=int)
    tram = nx.read_adjlist("data/tram.txt", create_using=nx.MultiGraph(), nodetype=int)

    # Add each edge to the final graph
    for edge in boat.edges():
        G.add_edge(*edge, type='boat')
    for edge in cart.edges():
        G.add_edge(*edge, type='cart')
    for edge in tram.edges():
        G.add_edge(*edge, type='tram')

    return G

def connections(graph: nx.Graph, node: int) -> np.ndarray[int]:
    """ Find the valid connections from one point ot another one, specifying also which 
        transport can be used to reach the destination node from the source.

    Args:
        graph (nx.Graph): The nodes' game graph.
        node  (int): The source node.

    Returns:
        _np.ndarray[int]: Encoding of all the connections from the source node.
    """
    connections = graph.edges(node, data=True)
    encoded_connections = []
    for source, dest, data in connections:
        transport_type = data["type"]
        encoded_connections.append([source, dest]+[1 if transport_type == t else 0 for t in ['boat', 'tram', 'cart']])

    return np.array(encoded_connections)


def node_one_hot_encoding(node_id: int, num_nodes: int) -> list[int]:
    """ Generate a one-hot encoding of the input node in the graph.

    Args:
        node_id (int): Input node.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        list[int]: The one-hot encoding.
    """
    return [1 if i == node_id - 1 else 0 for i in range(num_nodes)]