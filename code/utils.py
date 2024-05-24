import inspect
import numpy as np
from tigramite.toymodels import structural_causal_processes
from sklearn.metrics import confusion_matrix
from tigramite.toymodels.structural_causal_processes import links_to_graph

def tigramite_to_adjacency(graph, discard_self=True, discard_lags=True, discard_direction=False, **kwargs):
    """
    Returns the adjancency matrix for a tigramite graph. 

    Parameters
    ==========
    graph: array_like (n, n, tau_max + 1)
        tigramite graph for n nodes with maximum lag tau_max
    discard_self: bool
        whether to include self-adjacencies (default: false)
    discard_lags: bool
        whether to treat links with different lags between the same two nodes as equal (default: true) 
    discard_direction: bool
        whether to loose directionality (default: False)
    """
    adj = np.where((graph == "-->") | (graph == "o-o") | (graph == "x-x"), True, False)
    for i, j, tau in np.ndindex(*adj.shape):
        if i == j and discard_self:
            adj[i, j, tau] = False
        if i != j and discard_direction:
            adj[i, j, tau] = adj[i, j, tau] | adj[j, i, tau]
    if discard_lags:
        adj = np.where(np.any(adj, axis=2), 1, 0)

    return adj

def coupling_to_graph(d, return_val=True, **kwargs):

    # construct links
    n = d.shape[0]
    links = {i: [((j, -1), d[i, j], lambda x: x) for j in range(n) if d[i, j] != 0] for i in range(n)}

    graph = links_to_graph(links, **kwargs)

    if not return_val:
        return graph
    else:
        val = np.zeros_like(graph, dtype=float)
        val[:,:,1] = d.T
        return graph, val

def hamming(graph0, graph1, **kwargs):
    """
        Returns the Hamming distance between the adjacency matrices of to tigramite graphs. 
        Excluding a link is panalized with a value of one, flipping a link with two.

        Additional arguments are passed to the adjacency function.
    """
    
    return (tigramite_to_adjacency(graph0, **kwargs) ^ tigramite_to_adjacency(graph1, **kwargs)).sum()

def flatten(x, remove_self=True):
        return [x[idx] for idx in np.ndindex(x.shape) if idx[0] != idx[1] or not remove_self]

def accuracy_metrics(true, pred, **kwargs):
    
    # convert graphs into adjacency matrix
    true = tigramite_to_adjacency(true, **kwargs)
    pred = tigramite_to_adjacency(pred, **kwargs)

    remove_self = kwargs["remove_self"] if "remove_self" in kwargs else True
    
    tn, fp, fn, tp = confusion_matrix(flatten(true, remove_self), flatten(pred, remove_self), labels=[0, 1]).ravel()
    tpr = tp/(tp + fn) if tp + fn != 0 else np.nan
    fpr = fp/(fp + tn) if fp + tn != 0 else np.nan
    tnr = tn/(tn + fp) if tn + fp != 0 else np.nan
    fnr = fn/(fn + tp) if fn + tp != 0 else np.nan
    fsc = 2*tp/(2*tp + fn + fp) if 2*tp + fn + fp != 0 else np.nan

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr, "F-Score": fsc}

def screen(f, argsdict):
    return {key: value for key, value in argsdict.items() if key in inspect.signature(f).parameters.keys()}

# tests for the adjacency 
graph0 = structural_causal_processes.links_to_graph(structural_causal_processes.generate_structural_causal_process(3, 2, max_lag=1, seed=15)[0])
graph0 = np.concatenate((graph0, np.full((3, 3, 1), "")), axis=2) # to check the differing lags
graph1 = structural_causal_processes.links_to_graph(structural_causal_processes.generate_structural_causal_process(3, 1, max_lag=2, seed=15)[0])
empty  = np.full_like(graph0, "")

# graph vs. empy -> Hamming = number of links
# assert hamming(graph0, empty) == 2
# assert hamming(graph1, empty) == 1
# graph0 vs. graph1 without lags -> Hamming = 1
# assert hamming(graph0, graph1, discard_lags=True) == 1

# accuracy metrics for correct prediction
# assert accuracy_metrics(graph0, graph0) == (1.0, 0.0, 1.0, 0.0)

# accuracy metrics without lags
# assert accuracy_metrics(graph0, graph1) == (0.5, 0.0, 1.0, 0.5)

# acurracy metrics with lags
# assert accuracy_metrics(graph0, graph1, discard_lags=False) == (0.0, 1/16, 15/16, 1.0)

# discarding direction and lags -> Should have same Hamming distance
# assert hamming(graph0, graph1, discard_lags=True, discard_direction=True) == 0

def expand_mask(mask, N=1):
    """Expands the boolean array mask to be used with tigramite.
       All variables are masked equally.

    Parameters
    ----------
    mask: array-like (T)
        mask array 
    N : int, optional
        number of variables, default 1
    """
    return np.expand_dims(np.tile(mask, (N, 1)).T, axis=0)


import matplotlib.pyplot as plt

def scale_font(scaling=1):
    axs = plt.gcf().get_axes()
    for ax in axs:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax._children):
            if hasattr(item, "set_fontsize"): item.set_fontsize(scaling*item.get_fontsize())
plt.scale_font = scale_font


import scipy as sp

def griddata(x, y, z, N=100, **kwargs):
    # extent
    mx = np.min(x)
    Mx = np.max(x)
    my = np.min(y)
    My = np.max(y)

    # grid
    X = np.linspace(mx, Mx, N, endpoint=True)
    Y = np.linspace(my, My, N, endpoint=True)
    X, Y = np.meshgrid(X, Y)
    print(X.shape, Y.shape)

    # interpolate
    z = sp.interpolate.griddata((x, y), z, (X, Y), **screen(sp.interpolate.griddata, kwargs))
    
    # plot 
    plt.imshow(z, extent=(mx, Mx, my, My), origin="lower", **screen(plt.imshow, kwargs))

plt.griddata = griddata