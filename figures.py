import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import Bbox

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

def generations_plot(ax, generations, xkey, ykey, **kwargs):

    show = kwargs.get('show', None)
    interval = kwargs.get('interval', 1)
    rank = kwargs.get('rank', np.inf)
    cmap = kwargs.get('cmap', np.inf)
    zorder = kwargs.get('zorder', 0)
    initial = kwargs.get('initial', False)

    artists = []
    labels = []

    gen = generations[0]

    x, y = np.array(
        [[s['fitness'][xkey], s['fitness'][ykey]] for i, s in gen.items()]
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

    color = '#cccccc'

    if initial:

        labels.append(f'Initial Population')

        anc = ax.scatter(
            x[non_compliant_indices] / 1e6, y[non_compliant_indices] / 1e3,
            ec = 'none', lw = .1,
            fc = color, zorder = zorder, marker = 'X',
        )

        ac = ax.scatter(
            x[compliant_indices] / 1e6, y[compliant_indices] / 1e3,
            ec = 'none', lw = .1,
            fc = color, zorder = zorder + 1,
        )

        ap = ax.scatter(
            x[pareto_indices] / 1e6, y[pareto_indices] / 1e3,
            ec = 'none', lw = .1,
            fc = color, zorder = zorder + 2, marker = 's'
        )

        artists.append((anc, ac, ap))

    n = max(generations.keys())

    if show is None:

        show = list(range(interval, n + 1, interval)) + [max(generations.keys()) - 1]
        cont = True

    else:

        cont = False

    ax.dataLim = Bbox.null()

    for jdx, idx in enumerate(show):

        idx = int(idx)
        gen = generations[idx]

        x, y = np.array(
            [[s['fitness'][xkey], s['fitness'][ykey]] for i, s in gen.items()]
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

        color = cmap(idx) if cont else cmap(jdx)

        labels.append(f'Generation {idx}')

        anc = ax.scatter(
            x[non_compliant_indices] / 1e6, y[non_compliant_indices] / 1e3,
            ec = 'k', lw = .5,
            fc = color, zorder = zorder, marker = 'X'
        )

        ac = ax.scatter(
            x[compliant_indices] / 1e6, y[compliant_indices] / 1e3,
            ec = 'k', lw = .5,
            fc = color, zorder = zorder + 1,
        )

        ap = ax.scatter(
            x[pareto_indices] / 1e6, y[pareto_indices] / 1e3,
            ec = 'k', lw = .5,
            fc = color, zorder = zorder + 2, marker = 's'
        )

        artists.append((anc, ac, ap))

        zorder += 3

    if cont:

        sm = matplotlib.cm.ScalarMappable(
            cmap = cmap.cmap, norm = cmap.norm
        )

        _ = plt.colorbar(sm, ax = ax, label = 'Generation')

    else:

        _ = ax.legend(
            artists, labels,
            handler_map = {tuple: HandlerTuple(ndivide = None)},
            loc = 1, framealpha = 1,
            )

    return ax

def vehicles_scatter_pie(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]

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
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]

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
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]

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

    loc = np.array(list(range(len(x)))) + .5
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

def ports_stacked_bar(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]

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

    indices = np.argsort(x)
    b = {k: np.array(v)[indices] for k, v in p.items()}

    loc = np.array(list(range(len(x)))) + .5
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

def depots_bar(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank) and p['compliant']]

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
        for k in keys:

            p[k].append(len(s['vehicles'].get(k, [])))

    indices = np.argsort(x)
    b = {k: np.array(v)[indices] for k, v in p.items()}

    loc = np.array(list(range(len(x)))) + .5
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

def vehicles_scatter_pie_int(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]
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

    xi = (x - x.min()) / (x.max() - x.min())
    yi = (y - y.min()) / (y.max() - y.min())

    dist = np.cumsum(np.sqrt(np.array(
        [0.] + [(xi[i] - xi[i - 1]) ** 2 + (yi[i] - yi[i - 1]) ** 2 \
        for i in range(1, len(x))]
        )))

    d = np.linspace(0, dist.max(), n_shown)

    xp = np.interp(d, dist, x)
    yp = np.interp(d, dist, y)

    p = {k: np.interp(xp, x, v) for k, v in p.items()}

    pl = np.array(list(p.values())).T

    s = np.array(
        [sum(pli) ** exponent for pli in pl],
        dtype = float
        )
    s /= s.mean()
    s *= size

    # print(s)

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
    
    _ = src.plot.plot_scatter_pie(ax, xp / 1e6, yp / 1e3, pl, s, **kw)
    
    xmean = (xp.mean() + (xp.max() - xp.mean()) * .1) / 1e6
    ymean = (yp.mean() + (yp.max() - yp.mean()) * .1) / 1e3
    
    kw = {
        'ha': 'left',
        'va': 'bottom',
        'bbox': {
            'fc': 'w',
            'lw': .1,
        },
    }

    _ = ax.text(
        xmean, ymean,
        fr"Largest: $\bf{{{int(pl.sum(axis = 1).max())}}}$ vehicles" + '\n'
        fr'Smallest: $\bf{{{int(pl.sum(axis = 1).min())}}}$ vehicles',
        **kw
        )
    
    return ax

def ports_scatter_pie_int(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]
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

    xi = (x - x.min()) / (x.max() - x.min())
    yi = (y - y.min()) / (y.max() - y.min())

    dist = np.cumsum(np.sqrt(np.array(
        [0.] + [(xi[i] - xi[i - 1]) ** 2 + (yi[i] - yi[i - 1]) ** 2 \
        for i in range(1, len(x))]
        )))

    d = np.linspace(0, dist.max(), n_shown)

    xp = np.interp(d, dist, x)
    yp = np.interp(d, dist, y)

    p = {k: np.interp(xp, x, v) for k, v in p.items()}

    pl = np.array(list(p.values())).T


    s = np.array(
        [sum(pli) ** exponent for pli in pl],
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

    _ = src.plot.plot_scatter_pie(ax, xp / 1e6, yp / 1e3, pl, s, **kw)

    xmean = (xp.mean() + (xp.max() - xp.mean()) * .1) / 1e6
    ymean = (yp.mean() + (yp.max() - yp.mean()) * .1) / 1e3
    
    kw = {
        'ha': 'left',
        'va': 'bottom',
        'bbox': {
            'fc': 'w',
            'lw': .1,
        },
    }

    _ = ax.text(
        xmean, ymean,
        fr"Largest: $\bf{{{int(pl.sum(axis = 1).max())}}}$ ports" + '\n'
        fr'Smallest: $\bf{{{int(pl.sum(axis = 1).min())}}}$ ports',
        **kw
        )
    
    # kw = {
    #     'xlim'
    # }
    
    return ax

def vehicles_stacked_bar_int(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]
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
    
    xi = (x - x.min()) / (x.max() - x.min())
    yi = (y - y.min()) / (y.max() - y.min())

    dist = np.cumsum(np.sqrt(np.array(
        [0.] + [(xi[i] - xi[i - 1]) ** 2 + (yi[i] - yi[i - 1]) ** 2 \
        for i in range(1, len(x))]
        )))

    d = np.linspace(0, dist.max(), n_shown)

    xp = np.interp(d, dist, x)
    yp = np.interp(d, dist, y)

    p = {k: np.interp(xp, x, v) for k, v in p.items()}

    xp /= 1e6

    bottom = np.zeros(len(xp))

    colors = cmap.colors(np.linspace(0, 1, len(p)))

    for i, (k, v) in enumerate(p.items()):

        _ = ax.fill_between(
            xp, v + bottom, bottom,
            label = k, color = colors[i], ec = 'k', lw = .25
            )

        bottom += v

    _ = ax.legend()
    
    
    kw = {
        'xlim': (xp.min(), xp.max()),
        'ylim': (0, bottom.max())
    }

    _ = ax.set(**kw)

    return ax

def ports_stacked_bar_int(ax, population, **kwargs):

    n_shown = kwargs.get('n_shown', 30)
    size = kwargs.get('size', .35e-1)
    cmap = kwargs.get('cmap', src.plot.Colormap())
    exponent = kwargs.get('exponent', 1)
    keys = kwargs.get('keys', [])
    rank = kwargs.get('rank', 0)

    solutions = [p for p in population.values() if (p['rank'] <= rank)]
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

    xi = (x - x.min()) / (x.max() - x.min())
    yi = (y - y.min()) / (y.max() - y.min())

    dist = np.cumsum(np.sqrt(np.array(
        [0.] + [(xi[i] - xi[i - 1]) ** 2 + (yi[i] - yi[i - 1]) ** 2 \
        for i in range(1, len(x))]
        )))

    d = np.linspace(0, dist.max(), n_shown)

    xp = np.interp(d, dist, x)
    yp = np.interp(d, dist, y)

    p = {k: np.interp(xp, x, v) for k, v in p.items()}

    xp /= 1e6

    bottom = np.zeros(len(xp))

    colors = cmap.colors(np.linspace(0, 1, len(p)))

    for i, (k, v) in enumerate(p.items()):

        _ = ax.fill_between(
            xp, v + bottom, bottom,
            label = k, color = colors[i], ec = 'k', lw = .25
            )

        bottom += v

    _ = ax.legend()

    # print(xp)

    kw = {
        'xlim': (xp.min(), xp.max()),
        'ylim': (0, bottom.max())
    }

    _ = ax.set(**kw)

    return ax
