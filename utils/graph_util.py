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