import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

import src as src

style = {
    'rcparams': {
        'font.size': 11,
        'font.family': 'serif'
    },
    'save': {
        'bbox_inches': 'tight',
        'dpi': 300,
    },
}

def generations_plot(ax, generations, interval = 10, figsize = (6, 6), rank = np.inf):

    matplotlib.rcParams.update(**style['rcparams'])

    n = len(generations)
    cmap = src.plot.Colormap()

    for idx, gen in generations.items():

        idx = int(idx)

        if not idx in list(range(0, n + 1, interval)) + [len(generations) - 1]:

            continue

        x, y = np.array(
            [list(s['fitness'].values()) for i, s in gen.items()]
        ).T

        compliant_indices = [
            i for i, s in enumerate(gen.values()) \
            if s['compliant'] and (1 <= s['rank'] <= rank) \
            ]
        pareto_indices = [
            i for i, s in enumerate(gen.values()) \
            if s['compliant'] and (s['rank'] == 0) \
            ]
        non_compliant_indices = [
            i for i, s in enumerate(gen.values()) \
            if not s['compliant'] and (s['rank'] <= rank) \
            ]

        _ = ax.scatter(
            x[compliant_indices] / 1e6, y[compliant_indices] / 1e3,
            label = f'Generation {idx}', ec = 'k', lw = .5,
            fc = cmap([idx / n]), zorder = 2,
        )

        _ = ax.scatter(
            x[pareto_indices] / 1e6, y[pareto_indices] / 1e3,
            label = f'Generation {idx}', ec = 'k', lw = .5,
            fc = cmap([idx / n]), zorder = 3, marker = 's'
        )

        _ = ax.scatter(
            x[non_compliant_indices] / 1e6, y[non_compliant_indices] / 1e3,
            label = f'Generation {idx}', ec = 'k', lw = .5,
            fc = cmap([idx / n]), zorder = 1, marker = 'X'
        )

    return ax

def vehicles_scatter_pie(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])

    solutions = [p for p in population.values() if (p['rank'] <= 0) and p['compliant']]

    indices = src.optimization.n_highest_spread(
        [list(s['fitness'].values()) for s in solutions], n_shown,
    )
    solutions = [solutions[i] for i in indices]

    indices = np.argsort([list(s['fitness'].values())[0] for s in solutions])
    solutions = [solutions[i] for i in indices]

    x, y = np.array(
            [list(s['fitness'].values()) for s in solutions]
        ).T
    
    p = {k: [] for k in keys}

    for s in solutions:

        types = np.array([v for d in s['vehicles'].values() for v in d])

        for k, v in p.items():

            v.append(sum(types == k))

    pl = np.array(list(p.values())).T

    s = np.array(
        [sum([len(v) for v in si['vehicles'].values()]) ** exponent for si in solutions],
        dtype = float
        )
    s /= s.mean()
    s *= size

    kw = {
        'pad': 0.005,
        'patch': {
            'edgecolor': 'k',
            'linewidth': .35,
            'zorder': 2,
        },
        'labels': list(p.keys()),
        'colors': cmap.colors(np.linspace(0, 1, len(p))),
    }

    _ = src.plot.plot_scatter_pie(ax, x / 1e6, y / 1e3, pl, s, **kw)
    
    return ax

def ports_scatter_pie(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])

    solutions = [p for p in population.values() if (p['rank'] <= 0) and p['compliant']]

    indices = src.optimization.n_highest_spread(
        [list(s['fitness'].values()) for s in solutions], n_shown,
    )
    solutions = [solutions[i] for i in indices]

    indices = np.argsort([list(s['fitness'].values())[0] for s in solutions])
    solutions = [solutions[i] for i in indices]

    x, y = np.array(
            [list(s['fitness'].values()) for s in solutions]
        ).T
    
    p = {k: [] for k in keys}

    for s in solutions:

        types = np.array([v for d in s['ports'].values() for v in d])

        for k, v in p.items():

            v.append(sum(types == k))

    # p = defaultdict(lambda: {i: 0 for i in range(len(solutions))})

    # for i, s in enumerate(solutions):

    #     types = np.array([v for d in s['ports'].values() for v in d])

    #     u, uc = np.unique(types, return_counts = True)

    #     for j, u in enumerate(u):

    #         p[u][i] = uc[j]

    # p = {k: list(v.values()) for k, v in p.items()}
    # keys = list(p.keys())
    # keys.sort()
    # p = {k: p[k] for k in keys}
    # # print(p)

    pl = np.array(list(p.values())).T

    s = np.array(
        [sum([len(v) for v in si['ports'].values()]) ** exponent for si in solutions],
        dtype = float
        )
    s /= s.mean()
    s *= size

    kw = {
        'pad': 0.005,
        'patch': {
            'edgecolor': 'k',
            'linewidth': .35,
            'zorder': 2,
        },
        'labels': list(p.keys()),
        'colors': cmap.colors(np.linspace(0, 1, len(p))),
    }

    _ = src.plot.plot_scatter_pie(ax, x / 1e6, y / 1e3, pl, s, **kw)
    
    return ax

def vehicles_stacked_bar(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])

    solutions = [p for p in population.values() if (p['rank'] <= 0) and p['compliant']]

    if n_shown  < len(solutions):

        indices = src.optimization.n_highest_spread(
            [list(s['fitness'].values()) for s in solutions], n_shown,
        )
        solutions = [solutions[i] for i in indices]

    indices = np.argsort([list(s['fitness'].values())[0] for s in solutions])
    solutions = [solutions[i] for i in indices]

    x, y = np.array(
            [list(s['fitness'].values()) for s in solutions]
        ).T

    p = {k: [] for k in keys}

    for s in solutions:

        types = np.array([v for d in s['vehicles'].values() for v in d])

        for k, v in p.items():

            v.append(sum(types == k))

    indices = np.argsort(x)
    b = {k: np.array(v)[indices] for k, v in p.items()}

    loc = list(range(len(x)))
    bottom = np.zeros(len(x))

    colors = cmap.colors(np.linspace(0, 1, len(p)))

    for i, (k, v) in enumerate(b.items()):

        _ = ax.bar(
            loc, height = v, bottom = bottom, zorder = 2,
            label = k, color = colors[i], ec = 'k', lw = .25, width = 1,
        )

        bottom += v

    _ = ax.legend()

    return ax

def vehicles_ports_bar(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])

    solutions = [p for p in population.values() if (p['rank'] <= 0) and p['compliant']]

    if n_shown  < len(solutions):

        indices = src.optimization.n_highest_spread(
            [list(s['fitness'].values()) for s in solutions], n_shown,
        )
        solutions = [solutions[i] for i in indices]

    indices = np.argsort([list(s['fitness'].values())[0] for s in solutions])
    solutions = [solutions[i] for i in indices]

    x, y = np.array(
            [list(s['fitness'].values()) for s in solutions]
        ).T

    p = {k: [] for k in keys}

    for s in solutions:

        types = np.array([v for d in s['ports'].values() for v in d])

        for k, v in p.items():

            v.append(sum(types == k))

    # p = defaultdict(lambda: {i: 0 for i in range(len(solutions))})

    # for i, s in enumerate(solutions):

    #     types = np.array([v for d in s['ports'].values() for v in d])

    #     u, uc = np.unique(types, return_counts = True)

    #     for j, u in enumerate(u):

    #         p[u][i] = uc[j]

    # p = {k: list(v.values()) for k, v in p.items()}
    # keys = list(p.keys())
    # keys.sort()
    # p = {k: p[k] for k in keys}

    indices = np.argsort(x)
    b = {k: np.array(v)[indices] for k, v in p.items()}

    loc = list(range(len(x)))
    bottom = np.zeros(len(x))

    colors = cmap.colors(np.linspace(0, 1, len(p)))

    for i, (k, v) in enumerate(b.items()):

        _ = ax.bar(
            loc, height = v, bottom = bottom, zorder = 2,
            label = k, color = colors[i], ec = 'k', lw = .25, width = 1,
        )

        bottom += v

    _ = ax.legend()

    return ax