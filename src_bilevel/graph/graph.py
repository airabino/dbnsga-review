'''
Module for handling of graphs.
'''

import json
import numpy as np
import pandas as pd
import networkx as nx

from scipy.spatial import KDTree

def cypher(graph):

    encoder = {k: idx for idx, k in enumerate(graph.nodes)}
    decoder = {idx: k for idx, k in enumerate(graph.nodes)}

    return encoder, decoder

# Functions for NLG JSON handling 

class NpEncoder(json.JSONEncoder):
    '''
    Encoder to allow for numpy types to be converted to default types for
    JSON serialization. For use with json.dump(s)/load(s).
    '''
    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        if isinstance(obj, np.floating):

            return float(obj)

        if isinstance(obj, np.ndarray):

            return obj.tolist()

        if isinstance(obj, np.bool_):

            return int(obj)

        return super(NpEncoder, self).default(obj)

def nlg_to_json(nlg, filename):
    '''
    Writes nlg to JSON, overwrites previous
    '''

    with open(filename, 'w') as file:

        json.dump(nlg, file, indent = 4, cls = NpEncoder)

def nlg_from_json(filename):
    '''
    Loads graph from nlg JSON
    '''

    with open(filename, 'r') as file:

        nlg = json.load(file)

    return nlg

# Functions for NetworkX graph .json handling

def graph_to_json(graph, filename, **kwargs):
    '''
    Writes graph to JSON, overwrites previous
    '''

    with open(filename, 'w') as file:

        json.dump(nlg_from_graph(graph, **kwargs), file, indent = 4, cls = NpEncoder)

def graph_from_json(filename, **kwargs):
    '''
    Loads graph from nlg JSON
    '''

    with open(filename, 'r') as file:

        nlg = json.load(file)

    return graph_from_nlg(nlg, **kwargs)

# Functions for converting between NLG and NetworkX graphs

def graph_from_nlg(nlg, **kwargs):

    return nx.node_link_graph(nlg, edges = 'links', **kwargs)

def nlg_from_graph(nlg, **kwargs):

    nlg = nx.node_link_data(nlg, edges = 'links', **kwargs)

    return nlg

# Functions for graph operations

def random_node(graph):

    source = np.random.choice(list(graph.nodes))

    return source, graph._node[source]

def random_edge(graph):

    edge = list(graph.edges)[np.random.randint(0, graph.number_of_edges() - 1)]

    source, target = edge[0], edge[1]

    return source, target, graph._adj[source][target]

def subgraph(graph, nodes, edges = True):

    _node = graph._node
    _adj = graph._adj

    node_list = [(n, _node[n]) for n in nodes]

    edge_list = []

    if edges:

        for source in nodes:
            for target in nodes:

                edge_list.append((source, target, _adj[source].get(target, None)))

        edge_list = [e for e in edge_list if e[2] is not None]

    subgraph = graph.__class__()

    subgraph.add_nodes_from(node_list)

    subgraph.add_edges_from(edge_list)

    subgraph.graph.update(graph.graph)

    return subgraph

def subgraph_edges(graph, edges):
    # print(sources + targets)

    _node = graph._node
    _adj = graph._adj

    keep_nodes = [e for edge in edges for e in edge]

    node_list = [(k, graph._node[k]) for k in keep_nodes]

    edge_list = [(s, t, _adj[s][t]) for s, t in edges]

    subgraph = graph.__class__()

    subgraph.add_nodes_from(node_list)

    subgraph.add_edges_from(edge_list)

    subgraph.graph.update(graph.graph)

    return subgraph

def supergraph(graphs):

    supergraph = graphs[0].__class__()

    nodes = []

    edges = []

    names = []

    show = True

    for graph in graphs:

        for source, adj in graph._adj.items():

            names.append(source)

            coords_s = (graph._node[source]['x'], graph._node[source]['y'])

            nodes.append((coords_s, graph._node[source]))

            for target, edge in adj.items():

                coords_t = (graph._node[target]['x'], graph._node[target]['y'])

                edges.append((coords_s, coords_t, edge))

    supergraph.add_nodes_from(nodes)

    supergraph.add_edges_from(edges)

    supergraph = nx.relabel_nodes(
        supergraph, {k: names[idx] for idx, k in enumerate(supergraph.nodes)}
        )

    return supergraph

def order_nodes(graph, ordered_nodes, **kw):

    nodes = [(s, graph._node[s]) for s in ordered_nodes]
    edges = [(s, t, e) for s, a in graph._adj.items() for t, e in a.items()]

    out = graph.__class__()
    out.add_nodes_from(nodes)
    out.add_edges_from(edges)

    return out

def sort_nodes(graph, field, **kw):

    keys = list(graph.nodes())
    values = [n.get(field, 0) for n in graph._node.values()]

    indices = np.argsort(values)

    ordered_nodes = [keys[i] for i in indices]

    return order_nodes(graph, ordered_nodes, **kw)

def giant_connected_component(graph):

    if nx.is_directed(graph):

        components = list(nx.components.strongly_connected_components(graph))

    else:

        components = list(nx.components.connected_components(graph))

    gcc_index = np.argmax([len(c) for c in components])

    remove = []

    for idx, component in enumerate(components):

        if idx == gcc_index:

            continue

        remove.extend(list(component))

    graph.remove_nodes_from(remove)

    return graph

def remove_self_edges(graph):

    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph