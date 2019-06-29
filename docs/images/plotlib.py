try:
    from matplotlib import pyplot as plt
    import matplotlib
except:
    import matplotlib
    matplotlib.rcParams['backend'] = 'TkAgg'
    from matplotlib import pyplot as plt
import numpy as np
import pdb

def cornertex(s, ax, offset=(0,0), fontsize=14):
    plt.text(0.02+offset[0],0.95+offset[1],s,transform=ax.transAxes,color='k',va='top',ha='left',fontsize=fontsize, zorder=100)

class DataPlt():
    '''
    Dynamic plot context, intended for displaying geometries.
    like removing axes, equal axis, dynamically tune your figure and save it.

    Args:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.

    Attributes:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.
        ax (Axes): matplotlib Axes instance.

    Examples:
        with DynamicShow() as ds:
            c = Circle([2, 2], radius=1.0)
            ds.ax.add_patch(c)
    '''

    def __init__(self, figsize=(6, 4), filename=None, dpi=300):
        self.figsize = figsize
        self.filename = filename
        self.ax = None
        self.fig = None

    def __enter__(self):
        _setup_mpl()
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.gca()
        return self

    def __exit__(self, *args):
        if self.filename is not None:
            print('Press `c` to save figure to "%s", `Ctrl+d` to break >>' %
                  self.filename)
            pdb.set_trace()
            plt.savefig(self.filename, dpi=300)
        else:
            pdb.set_trace()


class NoBoxPlt():
    '''
    Dynamic plot context, intended for displaying geometries.
    like removing axes, equal axis, dynamically tune your figure and save it.

    Args:
        figsize (tuple, default=(6,4)): figure size.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.

    Attributes:
        figsize (tuple, default=(6,4)): figure size.
        graph_layout (tuple|None): number of graphs, None for single graph.
        filename (filename, str): filename to store generated figure, if None, it will not save a figure.
        ax (Axes): matplotlib Axes instance.

    Examples:
        with DynamicShow() as ds:
            c = Circle([2, 2], radius=1.0)
            ds.ax.add_patch(c)
    '''

    def __init__(self, figsize=(6, 4), graph_layout=None, filename=None, dpi=300):
        self.figsize = figsize
        self.filename = filename
        self.ax = None
        self.graph_layout = graph_layout

    def __enter__(self):
        _setup_mpl()
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        if self.graph_layout is None:
            self.ax = plt.subplot(111)
        else:
            self.ax = []
            self.gs = plt.GridSpec(*self.graph_layout)
            for i in range(self.graph_layout[0]):
                for j in range(self.graph_layout[1]):
                    self.ax.append(plt.subplot(self.gs[i, j]))
        return self

    def __exit__(self, *args):
        axes = [self.ax] if self.graph_layout is None else self.ax
        for ax in axes:
            ax.axis('equal')
            ax.axis('off')
        plt.tight_layout()
        if self.filename is not None:
            print('Press `c` to save figure to "%s", `Ctrl+d` to break >>' %
                  self.filename)
            pdb.set_trace()
            plt.savefig(self.filename, dpi=300)
        else:
            pdb.set_trace()


def _setup_mpl():
    '''customize matplotlib.'''
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 18


def _setup_font():
    myfont = matplotlib.font_manager.FontProperties(
        family='wqy', fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    matplotlib.rcParams["pdf.fonttype"] = 42
    return myfont


def visualize_tree(pairs, geometry):
    if len(geometry)==2:
        xs, ys = np.meshgrid(np.arange(geometry[0]), np.arange(geometry[1]), indexing='ij')
    else:
        num_bit = geometry[0]
        t = np.linspace(0,2*np.pi*(num_bit-1)/num_bit,num_bit)
        xs, ys = np.cos(t), np.sin(t)
    locs = np.concatenate([xs[...,None], ys[...,None]], axis=-1).reshape([-1,2])
    plt.scatter(locs[:,0], locs[:,1], s=80, zorder=101)
    for i, loc in enumerate(locs):
        plt.text(loc[0], loc[1]-0.2, '%d'%i, fontsize=18, va='center', ha='center')
    wl = np.array([p[2] for p in pairs])
    w_interval = wl.max()-wl.min()
    wl/=w_interval*1.2
    wl-=wl.min()-0.01
    print(wl)
    for (i, j, _), w in zip(pairs, wl):
        start, end = locs[i], locs[j]
        cmap = plt.get_cmap('jet')
        plt.plot([start[0], end[0]], [start[1], end[1]],color=cmap(w*10))


def visualize_tree(pairs, geometry, engine='viznet', offsets=None):
    if len(geometry)==2:
        xs, ys = np.meshgrid(np.arange(geometry[0]), np.arange(geometry[1]), indexing='ij')
        num_bit = np.prod(geometry)
    else:
        num_bit = geometry[0]
        t = np.linspace(0,2*np.pi*(num_bit-1)/num_bit,num_bit)
        xs, ys = np.sqrt(num_bit)/2.5*np.cos(t), np.sqrt(num_bit)/2.5*np.sin(t)
    locs = np.concatenate([xs[...,None], ys[...,None]], axis=-1).reshape([-1,2])
    wl = np.log(np.array([p[2] for p in pairs]))
    if offsets is None:
        offsets = [(0,0)]*num_bit

    if engine == 'networkx':
        import networkx as nx
        G = nx.Graph()
        for i, loc in enumerate(locs):
            plt.text(loc[0], loc[1]-0.2, '%d'%i, fontsize=18, va='center', ha='center')
            G.add_node(i, loc=loc)
        G.add_edges_from([p[:2] for p in pairs])
        vmin = wl.min()-0.3
        vmax = wl.max()+0.3
        print(vmin, vmax)
        nx.draw(G, pos=locs, node_color='#A0CBE2', edge_color=np.log(wl), edge_vmin=vmin, edge_vmax=vmax,
                width=4, edge_cmap=plt.cm.Blues, with_labels=False, alpha=1)
    elif engine == 'matplotlib':
        for i, loc in enumerate(locs):
            plt.text(loc[0], loc[1]-0.2, '%d'%i, fontsize=18, va='center', ha='center')
        plt.scatter(xs, ys, s=200, zorder=101)
        w_interval = wl.max()-wl.min()
        wl/=w_interval*1.2
        wl-=wl.min()-0.01
        for (i, j, _), w in zip(pairs, wl):
            start, end = locs[i], locs[j]
            cmap = plt.get_cmap('hot')
            plt.plot([start[0], end[0]], [start[1], end[1]],color=cmap(w*10))
    else:
        import viznet
        cmap = plt.get_cmap('binary')
        w_interval = wl.max()-wl.min()
        wl=(wl-wl.min())/(w_interval+0.1)
        print(wl)
        wl+=0.1
        viznet.setting.node_setting['lw']=0

        node = viznet.NodeBrush('basic', size='small', color='#6699CC')
        nl = []
        for i, pos in enumerate(locs):
            ni = node >> np.array(pos)+offsets[i]
            ni.text(i)
            nl.append(ni)

        for (i,j,_), w in zip(pairs, wl):
            edge = viznet.EdgeBrush('-', color=cmap(w), lw=2)
            edge >> (nl[i], nl[j])
