'''
Module for computing adjacency for a graph via routing on another graph. An example
would be computing travel times between cities connected by highways or latency between
computers connected via the internet. Another case would be compting network distances
between all points in a subset of a greater network. In any case, the nodes of the former
network must be coincident, or nearly so, with nodes in the latter network.

In this module the graph for which adjacency is being computed will be referred to as the
"graph" while the graph on which the routing occurs will be referred to as the "atlas". In
cases where either could be used "graph" will be used as default.
'''

import numpy as np

from sys import maxsize

from scipy.spatial import KDTree
from itertools import count
from heapq import heappop, heappush

from ..progress_bar import ProgressBar

from .graph import graph_from_nlg, cypher
from ..utilities import pythagorean, haversine

def closest_nodes_from_coordinates(graph, x, y):
    '''
    Creates an assignment dictionary mapping between points and closest nodes
    '''

    if not hasattr(x, '__iter__'):

        x = [x]

    if not hasattr(y, '__iter__'):

        y = [y]

    nodes = list(graph.nodes)

    # Pulling coordinates from graph
    xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
    xy_graph = xy_graph.reshape((-1,2))

    # Creating spatial KDTree for assignment
    kd_tree = KDTree(xy_graph)

    # Shaping input coordinates
    xy_query = np.vstack((x, y)).T

    # Computing assignment
    result = kd_tree.query(xy_query)

    node_assignment = []

    for idx in range(len(x)):

        distance = result[0][idx]
        node = result[1][idx]

        node_assignment.append({
            'id': nodes[node],
            'query': xy_query[idx],
            'result': xy_graph[node],
            'distance': distance,
            })

    return node_assignment

def relate(atlas, graph):
    '''
    Creates an assignment dictionary mapping between points and closest nodes
    '''

    # Pulling coordinates from atlas
    id_atlas = list(atlas.nodes())
    xy_atlas = np.array([(n['x'], n['y']) for n in atlas._node.values()])
    xy_atlas = xy_atlas.reshape((-1,2))

    # Pulling coordinates from graph
    id_graph = list(graph.nodes())
    xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
    xy_graph = xy_graph.reshape((-1,2))

    # Creating spatial KDTree for assignment
    kd_tree = KDTree(xy_atlas)

    # Computing assignment
    result = kd_tree.query(xy_graph)

    node_assignment = []

    for idx in range(len(xy_graph)):

        distance = result[0][idx]
        node = result[1][idx]
        

        node_assignment.append(
                {
                    'id_atlas':id_atlas[node],
                    'id_graph':id_graph[idx],
                    'query':xy_graph[idx],
                    'result':xy_atlas[node],
                    'distance': haversine(*xy_graph[idx], *xy_atlas[node]),
                }
            )

    return node_assignment

def node_assignment(atlas, graph):

    x, y = np.array(
        [[val['x'], val['y']] for key, val in graph._node.items()]
        ).T

    graph_nodes = np.array(
        [key for key, val in graph._node.items()]
        ).T

    # print(graph_nodes)

    atlas_nodes = closest_nodes_from_coordinates(atlas, x, y)

    graph_to_atlas = (
        {graph_nodes[idx]: atlas_nodes[idx]['id'] for idx in range(len(graph_nodes))}
        )
    
    atlas_to_graph = {}

    for key, val in graph_to_atlas.items():

        if val in atlas_to_graph.keys():

            atlas_to_graph[val] += [key]

        else:

            atlas_to_graph[val] = [key]

    return graph_to_atlas, atlas_to_graph

def point_to_point(graph, origin_lonlat, destination_lonlats, **kwargs):

    objective = kwargs.get('objective', 'objective')
    fields = kwargs.get('fields', [])

    origin = closest_nodes_from_coordinates(graph, *origin_lonlat)[0]['id']

    destination_relations = closest_nodes_from_coordinates(
        graph, [d[0] for d in destination_lonlats], [d[1] for d in destination_lonlats],
        )

    destinations = [d['id'] for d in destination_relations]

    # print(origin, destinations)

    costs, values, paths = dijkstra(
        graph, [origin],
        objective = objective,
        fields = fields,
        )

    c = [costs[d] for d in destinations]
    v = [values[d] for d in destinations]
    p = [paths[d] for d in destinations]

    return c, v, p

def dijkstra(graph, origins, **kwargs):

    objective = kwargs.get('objective', 'objective')
    destination = kwargs.get('destination', '')
    fields = kwargs.get('fields', [])

    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths
    values = {}
    paths = {}

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        # Source is seen at the start of iteration and at 0 cost
        # costs[origin] = 0

        value = {f: graph._node.get(f, 0) for f in fields}
        path = [origin]

        heappush(
            heap, (graph._node[origin].get(objective, 0), next(c), value, path, origin)
            )

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, value, path, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost
        values[source] = value
        paths[source] = path

        if destination in costs:

            break

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_target < costs.get(target, np.inf)

            if savings:

                value_target = {k: v + edge.get(k, 0) for k, v in value.items()}

                path_target = path + [target]

                heappush(
                    heap, (cost_target, next(c), value_target, path_target, target)
                    )

    return costs, values, paths

def adjacency(atlas, graph, **kwargs):
    '''
    Adds adjacency to graph by routing on atlas
    '''
    objective = kwargs.get('objective', 'distance')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_depth = kwargs.get('maximum_depth', np.inf)
    fields = kwargs.get('fields', ['distance', 'time'])
    pb_kw = kwargs.get('progress_bar', {})
    depots = kwargs.get('depots', [])

    graph_to_atlas, atlas_to_graph = node_assignment(atlas, graph)

    destinations = list(graph.nodes)

    destinations_atlas = [graph_to_atlas[node] for node in destinations]

    for origin in ProgressBar(destinations, **pb_kw):

        origin_atlas = graph_to_atlas[origin]

        costs, values, paths = dijkstra(
            atlas,
            [origin_atlas],
            objective = objective,
            maximum_cost = np.inf if origin in depots else maximum_cost,
            maximum_depth = np.inf if origin in depots else maximum_depth,
            fields = fields,
            )

        adj = {}

        destinations_reached = np.intersect1d(
            list(values.keys()),
            destinations_atlas,
            )

        for destination in destinations_reached:

            nodes = atlas_to_graph[destination]

            for node in nodes:

                adj[node] = values[destination]
                # adj[node]['path'] = paths[destination]

        graph._adj[origin] = adj

    return graph

def route_edges(atlas, graph, **kwargs):
    '''
    Adds adjacency to graph by routing on atlas
    '''
    objective = kwargs.get('objective', 'distance')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    maximum_depth = kwargs.get('maximum_depth', np.inf)
    fields = kwargs.get('fields', ['distance', 'time'])
    pb_kw = kwargs.get('progress_bar', {})
    paths = kwargs.get('paths', False)

    graph_to_atlas, atlas_to_graph = node_assignment(atlas, graph)

    sources = list(graph.nodes)

    for source in ProgressBar(sources, **pb_kw):

        source_atlas = graph_to_atlas[source]

        targets = list(graph._adj[source].keys())

        costs, values, paths = dijkstra(
            atlas,
            [source_atlas],
            terminals = targets,
            objective = objective,
            maximum_cost = maximum_cost,
            maximum_depth = maximum_depth,
            fields = fields,
            )

        for target, edge in graph._adj[source].items():

            target_atlas = graph_to_atlas[target]

            for field in fields:

                edge[field] = values[target_atlas][field]

            if paths:

                edge['path'] = paths[target_atlas]

    return graph

def get_terminals(graph, origins, **kwargs):

    destinations = kwargs.get('destinations', [])
    objective = kwargs.get('objective', 'objective')
    return_paths = kwargs.get('return_paths', True)
    terminate_at_destinations = kwargs.get('terminate_at_destinations', True)
    maximum_cost = kwargs.get('maximum_cost', np.inf)


    nodes = graph._node
    edges = graph._adj

    costs = {}
    paths = {}

    terminal = {k: True for k in graph.nodes}

    terminals = []

    if terminate_at_destinations:

        terminals = [d for d in destinations if d not in origins]

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        paths[origin] = [origin]

        heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        if source in terminals:

            continue

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_target <= costs.get(target, np.inf)

            feasible = cost_target <= maximum_cost

            if savings & feasible:
               
                terminal[source] = False
                paths[target] = paths[source] + [target]

                heappush(heap, (cost_target, next(c), target))

    terminal = {k: terminal[k] for k in costs.keys()}

    return costs, paths, terminal

def reduction(atlas, origins = [], **kwargs):

    objective = kwargs.get('objective', 'distance')
    maximum_cost = kwargs.get('maximum_cost', np.inf)
    include_intersections = kwargs.get('include_intersections', True)
    snowball = kwargs.get('snowball', True)

    if include_intersections:

        intersections = []

        for source, adj in atlas._adj.items():

            if len(adj) != 2:

                intersections.append(source)

        origins += intersections

    heap = []
    c = count()

    for node in origins:

        heappush(heap, (next(c), node))

    _node = atlas._node

    nodes = []
    links = []

    while heap:

        idx, origin = heappop(heap)

        print(f'{idx} done, {len(heap)} in queue                 ', end = '\r')

        node = _node[origin]
        node['id'] = origin

        nodes.append(node)

        costs, paths, terminal = get_terminals(
            atlas,
            [origin],
            destinations = origins,
            objective = objective,
            maximum_cost = maximum_cost,
            terminate_at_destinations = True,
            return_paths = False,
            )

        terminal_nodes = [k for k, v in terminal.items() if v]

        destinations_reached = np.intersect1d(
            terminal_nodes,
            origins,
            )

        # print(len(destinations_reached))

        new_destinations = np.setdiff1d(
            terminal_nodes,
            origins,
            )

        for destination in destinations_reached:

            link = {}

            link['source'] = origin
            link['target'] = destination

            link[objective] = costs[destination]
            link['path'] = paths[destination]

            links.append(link)

        if snowball:
            for destination in new_destinations:

                heappush(heap, (next(c), destination))

                origins.append(destination)

    return graph_from_nlg({'nodes': nodes, 'links': links})