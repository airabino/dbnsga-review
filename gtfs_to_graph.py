import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import osmnx
import argparse

import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import src

def date(components):

    components = [int(c) for c in components.split('-')]
    
    return datetime(*components)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    type = str, dest = 'directory',
    help = 'The directory containing the GTFS feed',
    )
arg_parser.add_argument(
    '-d', '--date', type = date, dest = 'date',
    help = 'Date to be used to determine active services. Enter as YYYY-MM-DD.'
    )
arg_parser.add_argument(
    '-s', '--services', nargs = '*', type = str, dest = 'services',
    help = 'Active services to be used in graph creation. Overwritten by date.',
    )
arg_parser.add_argument(
    '-w', '--window', type = float, default = 7200., dest = 'window',
    help = 'Maximum time between trip end and subsequent trip start for an edge',
    )
arg_parser.add_argument(
    '-rr', '--remake_roadmap', action = 'store_true', dest = 'remake_roadmap',
    help = 'Remake the roadmap graph',
    )
arg_parser.add_argument(
    '-rl', '--remake_locations', action = 'store_true', dest = 'remake_locations',
    help = 'Remake the locations graph',
    )
arg_parser.add_argument(
    '-rt', '--remake_trips', action = 'store_true', dest = 'remake_trips',
    help = 'Remake the trips graph',
    )

def active_schedules(df, date):

    weekday = date.strftime('%A').lower()

    df_a = df[df[weekday] == 1]

    if all(df['start_date'] > date):

        df_a = df_a[df_a['start_date'] == df_a['start_date'].min()]

    elif all(df['end_date'] < date):

        df_a = df_a[df_a['end_date'] == df_a['end_date'].max()]

    else:
        
        df_a = df_a[df_a['start_date'] <= date]
        df_a = df_a[df_a['end_date'] >= date]

    return [str(s) for s in df_a['service_id'].to_list()]

def main(**kwargs):

    directory = kwargs['directory']
    date = kwargs['date']
    services = kwargs['services']
    window = kwargs['window']
    remake_roadmap = kwargs['remake_roadmap']
    remake_locations = kwargs['remake_locations']
    remake_trips = kwargs['remake_trips']

    '''
    Loading GTFS info
    '''
    print()
    print('Loading GTFS Data')

    directory = directory[:-1] if directory[-1] == '/' else directory

    routes_df = pd.read_csv(f'{directory}/routes.txt')
    trips_df = pd.read_csv(f'{directory}/trips.txt')
    shapes_df = pd.read_csv(f'{directory}/shapes.txt')
    calendar_df = pd.read_csv(f'{directory}/calendar.txt')
    stops_df = pd.read_csv(f'{directory}/stop_times.txt')

    '''
    Loading depots
    '''

    if os.path.exists(f'{directory}/depots.txt'):

        depots_df = pd.read_csv(f'{directory}/depots.txt')

    else:

        print(
            'No depots.txt provided in GTFS folder. An artificial depot called ' +
            '"Depot" will be created at the mean stop location'
            )

        depots_df = pd.DataFrame(
            data = {
                'depot': ['Depot'],
                'longitude': [shapes_df['shape_pt_lon'].mean()],
                'latitude': [shapes_df['shape_pt_lat'].mean()],
                }
            )

    '''
    Processing Calendar
    '''

    if date:

        calendar_df['start_date'] = pd.to_datetime(
            calendar_df['start_date'], format = "%Y%m%d"
        )

        calendar_df['end_date'] = pd.to_datetime(
            calendar_df['end_date'], format = "%Y%m%d"
        )

        services = active_schedules(calendar_df, date)

    '''
    Processing Routes
    '''
    print('Processing GTFS Data')

    routes = {row['route_id']: row.to_dict() for idx, row in routes_df.iterrows()}

    '''
    Processing Shapes
    '''

    shapes = (
        {r: {
            'longitude': df_r['shape_pt_lon'].to_numpy(),
            'latitude': df_r['shape_pt_lat'].to_numpy()
        } \
         for r, df_r in shapes_df.groupby('shape_id')}
    )

    '''
    Processing stop times
    '''

    start_times = {}
    finish_times = {}
    route_durations = {}

    midnight = datetime.strptime('00:00:00', '%H:%M:%S')

    valid_trips = []

    for trip, df_s in stops_df.groupby('trip_id'):

        try:

            d0 = datetime.strptime(df_s['arrival_time'].iloc[0], '%H:%M:%S')
            d1 = datetime.strptime(df_s['departure_time'].iloc[-1], '%H:%M:%S')

        except ValueError:

            continue

        start_times[trip] = (d0 - midnight).seconds
        finish_times[trip] = (d1 - midnight).seconds
        route_durations[trip] = (d1 - d0).seconds

        valid_trips.append(trip)

    '''
    Processing Trips
    '''

    trips_df = trips_df[trips_df['trip_id'].apply(lambda t: t in valid_trips)]

    if services:

        trips_df = trips_df[trips_df['service_id'].apply(lambda s: str(s) in services)]

    trips_df['distance'] = trips_df['shape_id'].apply(
        lambda s: np.sum(
            src.utilities.haversine(
                shapes[s]['longitude'][:-1], shapes[s]['latitude'][:-1],
                shapes[s]['longitude'][1:], shapes[s]['latitude'][1:]
            )
        )
    )

    shapes_used = trips_df['shape_id'].unique()

    trips_df['duration'] = trips_df['trip_id'].apply(lambda s: route_durations[s])
    trips_df['start'] = trips_df['trip_id'].apply(lambda s: start_times[s])
    trips_df['finish'] = trips_df['trip_id'].apply(lambda s: finish_times[s])

    trips = {r['trip_id']: r for _, r in trips_df.iterrows()}

    '''
    Terminal locations
    '''

    start =  {
        i: (s['longitude'][0], s['latitude'][0]) \
        for i, s in shapes.items() if i in shapes_used
    }

    finish =  {
        i: (s['longitude'][-1], s['latitude'][-1]) \
        for i, s in shapes.items() if i in shapes_used
    }

    depot = {r['depot']: (r['longitude'], r['latitude']) for _, r in depots_df.iterrows()}

    terminal = {
        **{f"{k}_s": v for k, v in start.items()},
        **{f"{k}_f": v for k, v in finish.items()},
        **{k: v for k, v in depot.items()}
    }

    terminal_types = {
        **{f"{k}_s": 'stop' for k, v in start.items()},
        **{f"{k}_f": 'stop' for k, v in finish.items()},
        **{k: 'depot' for k, v in depot.items()}
    }

    '''
    Pulling the roadmap
    '''

    if os.path.exists(f'{directory}/roadmap.json') and (not remake_roadmap):

        print('Loading Roadmap')

        roadmap = src.graph.graph_from_json(f'{directory}/roadmap.json')

    else:

        print('Building Roadmap')

        all_x = np.array([xi for s in shapes.values() for xi in s['longitude']])
        all_y = np.array([xi for s in shapes.values() for xi in s['latitude']])

        xmax, xmin, ymax, ymin = all_x.max(), all_x.min(), all_y.max(), all_y.min()

        buffer = .5

        bbox = (
            xmin - (xmax - xmin) * buffer,
            ymin - (ymax - ymin) * buffer,
            xmax + (xmax - xmin) * buffer,
            ymax + (ymax - ymin) * buffer,
        )

        roadmap_data = osmnx.graph.graph_from_bbox(
            bbox, network_type = 'drive', simplify = True, retain_all = False
            )
        roadmap_data = osmnx.routing.add_edge_speeds(roadmap_data)
        roadmap_data = osmnx.routing.add_edge_travel_times(roadmap_data)

        roadmap = nx.DiGraph(roadmap_data)

        for s, t, e in roadmap.edges(data = True):

            new_e = {
                'distance': e['length'],
                'time': e['travel_time'],
            }

            roadmap._adj[s][t] = new_e

        roadmap = src.graph.giant_connected_component(roadmap)

        src.graph.graph_to_json(roadmap, f'{directory}/roadmap.json')

    '''
    Building Locations graph
    '''

    if os.path.exists(f'{directory}/locations.json') and (not remake_locations):

        print('Loading Locations graph')

        locations = src.graph.graph_from_json(f'{directory}/locations.json')

    else:

        print('Building Locations graph')

        kw = {
            'objective': 'time',
            'fields': ['time', 'distance']
        }

        times = defaultdict(lambda: defaultdict(float))
        distances = defaultdict(lambda: defaultdict(float))

        targets = list(terminal.keys())
        target_locations = list(terminal.values())

        for source, location in tqdm(terminal.items()):

            _, values, _ = src.graph.point_to_point(
                roadmap, location, target_locations, **kw
            )

            for idx, target in enumerate(targets):

                times[source][target] = values[idx]['time']
                distances[source][target] = values[idx]['distance']

        times = dict(times)
        distances = dict(distances)

        locations = nx.DiGraph()
        locations.add_nodes_from(
            [(k, {'type': terminal_types[k], 'x': terminal[k][0], 'y': terminal[k][1]}) \
            for k, v in terminal.items()]
            )
        locations.add_edges_from(
            [(s, t, {'duration': times[s][t], 'distance': distances[s][t]}) \
            for s in targets for t in targets]
            )

        src.graph.graph_to_json(locations, f'{directory}/locations.json')

    '''
    Building Trips graph
    '''

    if os.path.exists(f'{directory}/trips.json') and (not remake_trips):

        print('Loading Trips graph')

        trips = src.graph.graph_from_json(f'{directory}/trips.json')

    else:

        print('Building Trips graph')

        nodes = []
        edges = []

        fields = ['distance', 'duration', 'start', 'finish']

        for s, source in tqdm(trips.items()):

            n_s = f'{source['shape_id']}_s'
            n_f = f'{source['shape_id']}_f'

            nodes.append(
                (s,
                 {'start_location': n_s, 'finish_location': n_f,
                  'route': source['route_id']} | \
                 {f: source[f] for f in fields}))

            for t, target in trips.items():
                
                n_t = f'{target['shape_id']}_s'

                transit = locations._adj[n_s][n_t]['duration']

                start_feasible = (
                    target['start'] >= source['finish'] + transit
                )
                finish_feasible = target['start'] <= source['finish'] + transit + window

                if start_feasible and finish_feasible:

                    edges.append((s, t))

        trips_graph = nx.DiGraph()
        trips_graph.add_nodes_from(nodes)
        trips_graph.add_edges_from(edges)

        src.graph.graph_to_json(trips_graph, f'{directory}/trips.json')

    print()

if __name__ == "__main__":

    args = sys.argv[1:]

    main(**vars(arg_parser.parse_args(args)))

    
