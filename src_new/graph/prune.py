import numpy as np
import networkx as nx

def trip_edge_weights(trips, locations):

    for s, t, e in trips.edges(data = True):

        source = trips._node[s]['finish_location']
        target = trips._node[t]['start_location']

        e['distance'] = locations._adj[source][target]['distance']
        e['duration'] = locations._adj[source][target]['duration']

        source = trips._node[s]['finish']
        target = trips._node[t]['start']

        e['elapsed_time'] = target - source

    return trips

def prune_by_value(trips, locations, field, lb = 0, ub = np.inf):

    drop = []

    for s, t, e in trips.edges(data = True):

        value = e[field]
        
        if (value < lb) or (value > ub):

            drop.append((s, t))

    trips.remove_edges_from(drop)

    return trips

def prune_by_portion(trips, locations, field, lb = 0, ub = np.inf):

    drop = []

    for s, t, e in trips.edges(data = True):

        value = trips._node[s][field]

        portion = e[field] / value
        
        if (portion < lb) or (portion > ub):

            drop.append((s, t))

    trips.remove_edges_from(drop)

    return trips

def prune_by_time_buffer(trips, locations, buffer = 0.):

    drop = []

    for s, t, e in trips.edges(data = True):

        duration = e['duration'] + trips._node[s]['duration'] * buffer
        
        if duration > e['elapsed_time']:

            drop.append((s, t))

    trips.remove_edges_from(drop)

    return trips