import os
import sys
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from operator import sub
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon, Wedge

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from itertools import cycle
# from shapely import Polygon

default_colors = matplotlib.rcParamsDefault['axes.prop_cycle'].by_key()['color'].copy()

class Colormap():

    color_schemes = {
        'day_night': ["#e6df44", "#f0810f", "#063852", "#011a27"],
        'beach_house': ["#d5c9b1", "#e05858", "#bfdccf", "#5f968e"],
        'autumn': ["#db9501", "#c05805", "#6e6702", "#2e2300"],
        'ocean': ["#003b46", "#07575b", "#66a5ad", "#c4dfe6"],
        'forest': ["#7d4427", "#a2c523", "#486b00", "#2e4600"],
        'aqua': ["#004d47", "#128277", "#52958b", "#b9c4c9"],
        'field': ["#5a5f37", "#fffae1", "#524a3a", "#919636"],
        'misty': ["#04202c", "#304040", "#5b7065", "#c9d1c8"],
        'greens': ["#265c00", "#68a225", "#b3de81", "#fdffff"],
        'citroen': ["#b38540", "#563e20", "#7e7b15", "#ebdf00"],
        'blues': ["#1e1f26", "#283655",  "#4d648d", "#d0e1f9"],
        'dusk': ["#363237", "#2d4262", "#73605b", "#d09683"],
        'ice': ["#1995ad", "#a1d6e2", "#bcbabe", "#f1f1f2"],
        'csu': ["#1e4d2b", "#c8c372"],
        'ucd': ['#022851', '#ffbf00'],
        'grayscale': ['#ffffff', '#000000'],
        'incose': ["#f2606b", "#ffdf79", "#c6e2b1", "#509bcf"],
        'sae': ["#01a0e9", "#005195", "#cacac8", "#9a9b9d", "#616265"],
        'trb': ["#82212a", "#999999", "#181818"],
        'ibm': ['#648fff', '#785ef0', '#dc267f', '#fe6100', '#ffb000'],
        'default_colors': default_colors,
    }

    def __init__(self, colors = 'viridis', **kwargs):

        self.build(colors, **kwargs)
        
    def __call__(self, values):

        return self.colors(values)

    def build(self, colors, **kwargs):

        self.vmin = kwargs.get('vmin', 0)
        self.vmax = kwargs.get('vmax', 1)
        self.flip = kwargs.get('flip', False)

        self.norm = matplotlib.colors.Normalize(self.vmin, self.vmax)

        if type(colors) == str:

            if colors in self.color_schemes.keys():

                colors_list = self.color_schemes[colors]

                if self.flip:

                    colors_list = np.flip(colors_list)

                self.cmap = LinearSegmentedColormap.from_list(
                    'custom', colors_list, N = 256)

            else:

                if self.flip:

                    colors += '_r'

                self.cmap = plt.colormaps.get_cmap(colors)

        else:

            if self.flip:

                colors = np.flip(colors)

            self.cmap = LinearSegmentedColormap.from_list(
                'custom', colors, N = 256)

    def categorical(self, values, **kwargs):

        u = np.unique(values)
        n = len(u)

        mapping = {ui: self.cmap(i / n) for i, ui in enumerate(u)}

        return np.array([mapping[v] for v in values])

    def colors(self, values, **kwargs):

        values = np.asarray(values).astype(float)
        # print(values)

        if isinstance(np.atleast_1d(values)[0], str):

            return self.categorical(values, **kwargs)

        values[values == np.inf] = np.nan
        values[values == -np.inf] = np.nan

        vmin = np.nanmin(values) if self.vmin is None else self.vmin
        vmax = np.nanmax(values) if self.vmax is None else self.vmax

        if vmin == vmax:

            values_norm = values

        else:

            values_norm = (
                (values - vmin) / (vmax - vmin)
                )

        self.norm = matplotlib.colors.Normalize(vmin, vmax)

        return self.cmap(values_norm)

def plot_edges(graph, ax, **kwargs):
    # print('s')

    _node = graph._node
    _adj = graph._adj

    cmap = kwargs.get('cmap', Colormap('viridis'))
    field = kwargs.get('field', None)
    scale = kwargs.get('scale', None)
    selection = kwargs.get('selection', None)
    colorbar = kwargs.get('colorbar', None)
    kw = kwargs.get('plot', {})

    if selection is None:

        selection = [(s, t) for s, adj in graph._adj.items() for t in adj.keys()]

    lines = []
    values = []
    widths = []

    for edge in selection:

        source, target = edge

        lines.append(np.array([
            [_node[source]['x'], _node[source]['y']],
            [_node[target]['x'], _node[target]['y']]
            ]))

        values.append(_adj[source][target].get(field, np.nan))
        widths.append(_adj[source][target].get(scale, None))

    lines = np.array(lines)
    values = np.array(values)
    widths = np.array(widths)

    indices = np.argsort(values)

    values = values[indices]
    lines = lines[indices]
    widths = widths[indices]

    # print(widths)

    if field is not None:

        kw['color'] = cmap(values)

    if scale is not None:
    
        kw['lw'] = widths

    edges_plot = LineCollection(lines, **kw)

    ax.add_collection(edges_plot)

    if colorbar is not None:

        plot_colorbar(cmap, ax, **colorbar)

    return edges_plot

def plot_nodes(graph, ax, **kwargs):

    cmap = kwargs.get('cmap', Colormap('viridis'))
    field = kwargs.get('field', None)
    scale = kwargs.get('scale', None)
    selection = kwargs.get('selection', None)
    colorbar = kwargs.get('colorbar', None)
    kw = kwargs.get('plot', {})

    if selection == []:

        return None

    if selection is None:

        selection = [s for s in graph.nodes]

    nodes = [graph._node[s] for s in selection]

    coords = np.array([[node['x'], node['y']] for node in nodes])

    sizes = np.array([v.get(scale, None) for v in nodes])

    colors = np.array([v.get(field, np.nan) for v in nodes])
    
    try:

        indices = np.argsort(colors)

    except TypeError as e:

        indices = list(range(len(colors)))

    colors = colors[indices]
    coords = coords[indices]
    sizes = sizes[indices]


    if field is not None:

        kw['c'] = cmap(colors)

    if scale is not None:

        kw['s'] = sizes

    # print(kw)

    nodes_plot = ax.scatter(
        coords[:, 0], coords[:, 1], **kw
        )

    if colorbar is not None:

        # plot_colorbar(nodes_plot, ax, **colorbar)
        plt.colorbar(nodes_plot, ax = ax, **colorbar)
    
    return nodes_plot

def plot_patches(graph, ax, **kwargs):

    cmap = kwargs.get('cmap', Colormap('viridis'))
    field = kwargs.get('field', None)
    patch = kwargs.get('patch', 'patch')
    selection = kwargs.get('selection', None)
    colorbar = kwargs.get('colorbar', None)
    kw = kwargs.get('plot', {})

    if selection == []:

        return None

    if selection is None:

        selection = [s for s in graph.nodes]

    nodes = [graph._node[s] for s in selection]

    values = np.array([v.get(field, np.nan) for v in nodes])

    if field is not None:

        kw['facecolors'] = cmap(values)

    patches = [graph._node[s][patch] for s in selection]

    nodes_plot = matplotlib.collections.PatchCollection(patches, **kw)
    ax.add_collection(nodes_plot)

    if colorbar is not None:

        plt.colorbar(nodes_plot, ax = ax, **colorbar)
    
    return nodes_plot

def qhull(x, y, **kwargs):

    q = kwargs.get('q', 21)
    r = kwargs.get('r', 1)
    kw = kwargs.get('patch', {})

    if not hasattr(r, '__iter__'):

        r = np.ones_like(x) * r

    n = len(x)

    theta = np.linspace(0, 2 * np.pi, q)

    x_out = np.zeros(n * q)
    y_out = np.zeros(n * q)

    #Generating the circular geometries
    for idx in range(n):
        x_out[q * idx:q * (idx + 1)] = x[idx] + np.cos(theta) * r[idx]
        y_out[q * idx:q * (idx + 1)] = y[idx] + np.sin(theta) * r[idx]

    hull = ConvexHull(np.vstack((x_out, y_out)).T)

    polygon = Polygon(hull._points[hull.vertices], **kw)

    return polygon

def plot_labels(graph, ax, **kwargs):

    field = kwargs.get('field', None)
    selection = kwargs.get('selection', None)
    kw = kwargs.get('plot', {})

    if selection == []:

        return None

    if selection is None:

        selection = [s for s in graph.nodes]

    nodes = [graph._node[s] for s in selection]

    x = np.array([node['x'] for node in nodes])
    y = np.array([node['y'] for node in nodes])

    # print(x)

    values = np.array([v.get(field, '') for v in nodes])

    indices = np.argsort(values)

    # values = values[indices]
    # x = x[indices]
    # y = y[indices]

    # print(y)

    for idx in indices:
        _ = ax.text(
            x[idx], y[idx], values[idx], **kw
            )

    return None

def plot_colorbar(cmap, ax, **kwargs):

    sm = matplotlib.cm.ScalarMappable(cmap = cmap.cmap, norm = cmap.norm)    
    sm.set_array([])

    colorbar = plt.colorbar(sm, ax = ax, **kwargs)

    return colorbar

def plot_graph(graph, ax, **kwargs):

    edges = kwargs.get('edges', None)
    nodes = kwargs.get('nodes', None)

    if edges is not None:

        edges_plot = plot_edges(graph, ax, **edges)

    else:

        edges_plot = None

    if nodes is not None:
       
        nodes_plot = plot_nodes(graph, ax, **nodes)

    else:

        nodes_plot = None


    return nodes_plot, edges_plot

class Norm():

    def __init__(self, x, **kwargs):

        self.x = x

        # Original limits
        self.x_min = min(x)
        self.x_max = max(x)

        # Normalized Data
        self.x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        self.x_norm -= np.mean(self.x_norm)

        self.x_norm_min = min(self.x_norm)
        self.x_norm_max = max(self.x_norm)

        self.scale = interp1d(
            [self.x_min, self.x_max], [self.x_norm_min, self.x_norm_max],
            fill_value = 'extrapolate'
            )

        self.rescale = interp1d(
            [self.x_norm_min, self.x_norm_max], [self.x_min, self.x_max],
            fill_value = 'extrapolate'
            )

    def scale1(self, y, buffer = 0):

        y_norm = (y - self.x_min) / (self.x_max - self.x_min)
        y_norm -= np.mean(self.x_norm)

        buffer = 1 + buffer

        y_scaled = np.interp(
            y, [self.x_min, self.x_max], [self.x_norm_min, self.x_norm_max]
            )

        return y_scaled

    def rescale1(self, y):

        y_scaled = np.interp(
            y, [self.x_norm_min, self.x_norm_max], [self.x_min, self.x_max]
            )

        return y_scaled

def plot_scatter_wedge(ax, x, y, p, s, **kwargs):

    patch_kw = kwargs.get('patch', {})

    pad = kwargs.get('pad', 0.05)

    # Initializing patch lists
    patches = []

    # Setting the scale
    x_norm = Norm(x)
    y_norm = Norm(y)
    x_normalized = x_norm.scale(x)
    y_normalized = y_norm.scale(y)


    n = len(x)

    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0

    for idx in range(n):

        portions = np.array(p[idx]) / sum(p[idx])

        thetas = 2 * np.pi * portions

        t0 = 0

        for jdx, theta in enumerate(thetas):

            xx = np.concatenate(
                (
                    [x_normalized[idx]],
                    x_normalized[idx] + np.cos(np.linspace(t0, t0 + theta, 100)) * s[idx],
                    [x_normalized[idx]],
                    )
                )

            yy = np.concatenate(
                (
                    [y_normalized[idx]],
                    y_normalized[idx] + np.sin(np.linspace(t0, t0 + theta, 100)) * s[idx],
                    [y_normalized[idx]],
                    )
                )

            xmin = min(xx) if min(xx) < xmin else xmin
            xmax = max(xx) if max(xx) > xmax else xmax
            ymin = min(yy) if min(yy) < ymin else ymin
            ymax = max(yy) if max(yy) > ymax else ymax

            xx = x_norm.rescale(xx)
            yy = y_norm.rescale(yy)

            xy = np.vstack((xx, yy)).T

            patch = Polygon(xy, fc = colors[jdx], ec = 'k', **patch_kw)
            patches[jdx].append(patch)

            t0 += theta

    for idx, p in enumerate(patches):

        collection = matplotlib.collections.PatchCollection(p, match_original = True)
        ax.add_collection(collection)

    ax.set(
        xlim = (
            x_norm.rescale(xmin) - pad * x_norm.rescale(xmax - xmin),
            x_norm.rescale(xmax) + pad * x_norm.rescale(xmax - xmin),
            ),
        ylim = (
            y_norm.rescale(ymin) - pad * y_norm.rescale(ymax - ymin),
            y_norm.rescale(ymax) + pad * y_norm.rescale(ymax - ymin),
            ),
        )

def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio

def plot_scatter_pie(ax, x, y, p, s, **kwargs):

    aspect_ratio = get_aspect(ax)

    patch_kw = kwargs.get('patch', {})
    colors = kwargs.get('colors', default_colors)
    labels = kwargs.get('labels', [None])

    pad = kwargs.get('pad', 0.05)

    # Initializing patch lists
    patches = [[] for _ in  range(len(p[0]))]

    # Setting the scale
    x_norm = Norm(x)
    y_norm = Norm(y)
    x_normalized = x_norm.scale(x)
    y_normalized = y_norm.scale(y)


    n = len(x)

    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0

    for idx in range(n):

        portions = np.array(p[idx]) / sum(p[idx])

        thetas = 2 * np.pi * portions

        t0 = 0

        color_cycle = cycle(colors)
        label_cycle = cycle(labels)

        for jdx, theta in enumerate(thetas):

            xx = np.concatenate(
                (
                    [x_normalized[idx]],
                    x_normalized[idx] + np.cos(np.linspace(t0, t0 + theta, 100)) *
                    s[idx] * aspect_ratio,
                    [x_normalized[idx]],
                    )
                )

            yy = np.concatenate(
                (
                    [y_normalized[idx]],
                    y_normalized[idx] + np.sin(np.linspace(t0, t0 + theta, 100)) * s[idx],
                    [y_normalized[idx]],
                    )
                )

            xmin = min(xx) if min(xx) < xmin else xmin
            xmax = max(xx) if max(xx) > xmax else xmax
            ymin = min(yy) if min(yy) < ymin else ymin
            ymax = max(yy) if max(yy) > ymax else ymax

            xx = x_norm.rescale(xx)
            yy = y_norm.rescale(yy)

            xy = np.vstack((xx, yy)).T

            kw = {
                **patch_kw,
                'fc': next(color_cycle),
                'label': next(label_cycle) if idx == 0 else None,
                # 'zorder': idx,
            }

            patch = Polygon(xy, **kw)

            ax.add_patch(patch)
            # patches[jdx].append(patch)

            t0 += theta

    # for idx, p in enumerate(patches):

    #     collection = matplotlib.collections.PatchCollection(p, match_original = True)
    #     ax.add_collection(collection)

    ax.set(
        xlim = (
            x_norm.rescale(xmin) - pad * x_norm.rescale(xmax - xmin),
            x_norm.rescale(xmax) + pad * x_norm.rescale(xmax - xmin),
            ),
        ylim = (
            y_norm.rescale(ymin) - pad * y_norm.rescale(ymax - ymin),
            y_norm.rescale(ymax) + pad * y_norm.rescale(ymax - ymin),
            ),
        )