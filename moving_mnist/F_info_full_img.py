import os
import time
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import seaborn as sns
from collections import defaultdict
from data import *
from models_conv import AE
from models import F_MLP,VAE

import numpy as np
plt.switch_backend('agg')

def _input(filename):
    prices = {}
    names = {}

    for line in open(filename).readlines():
        (name, src, dst, price) = line.rstrip().split()
        name = int(name.replace('M', ''))
        src = int(src.replace('C', ''))
        dst = int(dst.replace('C', ''))
        price = int(price)
        t = (src, dst)
        if t in prices and prices[t] <= price:
            continue
        prices[t] = price
        names[t] = name

    return prices, names


def _load(arcs, weights):
    g = {}
    for (src, dst) in arcs:
        if src in g:
            g[src][dst] = weights[(src, dst)]
        else:
            g[src] = {dst: weights[(src, dst)]}
    return g


def reverse(graph):
    r = {}

    for src in graph:
        for (dst, c) in graph[src].items():
            if dst in r:
                r[dst][src] = c
            else:
                r[dst] = {src: c}
    return r


def getCycle(n, g, visited=None, cycle=None):
    if visited is None:
        visited = set()
    if cycle is None:
        cycle = []
    visited.add(n)
    cycle += [n]
    if n not in g:
        return cycle
    for e in g[n]:
        if e not in visited:
            cycle = getCycle(e, g, visited, cycle)
    return cycle


def mergeCycles(cycle, G, RG, g, rg):
    allInEdges = []
    minInternal = None
    minInternalWeight = 100000

    # find minimal internal edge weight
    for n in cycle:
        for e in RG[n]:
            if e in cycle:
                if minInternal is None or RG[n][e] < minInternalWeight:
                    minInternal = (n, e)
                    minInternalWeight = RG[n][e]
                    continue
            else:
                allInEdges.append((n, e))

                # find the incoming edge with minimum modified cost
    minExternal = None
    minModifiedWeight = 0
    for s, t in allInEdges:
        u, v = rg[s].popitem()
        rg[s][u] = v
        w = RG[s][t] - (v - minInternalWeight)
        if minExternal is None or minModifiedWeight > w:
            minExternal = (s, t)
            minModifiedWeight = w

    u, w = rg[minExternal[0]].popitem()
    rem = (minExternal[0], u)
    rg[minExternal[0]].clear()
    if minExternal[1] in rg:
        rg[minExternal[1]][minExternal[0]] = w
    else:
        rg[minExternal[1]] = {minExternal[0]: w}
    if rem[1] in g:
        if rem[0] in g[rem[1]]:
            del g[rem[1]][rem[0]]
    if minExternal[1] in g:
        g[minExternal[1]][minExternal[0]] = w
    else:
        g[minExternal[1]] = {minExternal[0]: w}


# --------------------------------------------------------------------------------- #

def mst(root, G):
    """ The Chu-Lui/Edmond's algorithm
    arguments:
    root - the root of the MST
    G - the graph in which the MST lies
    returns: a graph representation of the MST
    Graph representation is the same as the one found at:
    http://code.activestate.com/recipes/119466/
    Explanation is copied verbatim here:
    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.
    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.
    """

    RG = reverse(G)

    if root in RG:
        RG[root] = {}
    g = {}
    for n in RG:
        if len(RG[n]) == 0:
            continue
        minimum = 100000
        s, d = None, None
        for e in RG[n]:
            if RG[n][e] < minimum:
                minimum = RG[n][e]
                s, d = n, e
        if d in g:
            g[d][s] = RG[s][d]
        else:
            g[d] = {s: RG[s][d]}

    cycles = []
    visited = set()
    for n in g:
        if n not in visited:
            cycle = getCycle(n, g, visited)
            cycles.append(cycle)

    rg = reverse(g)
    for cycle in cycles:
        if root in cycle:
            continue

        mergeCycles(cycle, G, RG, g, rg)

    return g

def adj2dict(mat):
    d = dict()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if i == j:
                continue
            if i not in d.keys():
                d[i] = {j: float(mat[i,j])}
            else:
                d[i].update( {j: float(mat[i, j])})

    return d

def dict2adj(d,num):


    mat = np.zeros((num,num))
    for i in d:
        for j in d[i]:
            mat[i,j]  = 1
    return mat


def main(args):
    ts = time.time()
    torch.cuda.set_device(args.device)


    def loss_fn(recon_x, x):

        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 64 * 64), x.view(-1, 64 * 64), size_average=False)
        return BCE/x.size(0)
    
    
    node_num = 5
    mat = np.ones((node_num,node_num)) * 100
    epoch_loss = 0.0

    for i in range(node_num):
        for j in range(node_num):
            if i == j:
                continue
            
            ae = AE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                conditional=args.conditional,
                num_labels=10 if args.conditional else 0).to(args.device)
            
            optimizer = torch.optim.Adam(ae.parameters(), lr=args.learning_rate)

            logs = defaultdict(list)
            train_dataset_pair = Moving_MNIST_Frame(i, j)
            train_loader_pair = torch.utils.data.DataLoader(dataset=train_dataset_pair, batch_size=64, shuffle=True)
            for epoch in range(args.epochs):

                epoch_loss = 0.0
                for iteration, (x, y) in enumerate(train_loader_pair):
                    x = x.to(args.device)
                    y = y.to(args.device)
                    x = x.unsqueeze(1)
                    mean_x, _ = ae(x)

                    loss = loss_fn(mean_x, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.data * x.size(0)

            print("{} -> {} ::  Loss: {}".format(i, j, epoch_loss / len(train_loader.dataset)))
            mat[i,j] = epoch_loss / len(train_loader.dataset)
    #np.save('mat.npy',mat)
    g = adj2dict(mat)
    print(g)
    min_tree = 1000000
    tree_dict = dict()
    
    for i in range(node_num):
        h = mst(int(i), g)
        val = 0.0
        for s in h:
            for k in h[s]:
                val += mat[s, k]
            
        if val < (min_tree - 0.01):
            min_tree = val
            tree_dict[0] = h
            print(i)
    print(tree_dict[0])
    T = dict2adj(tree_dict[0], node_num)
    print(T)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--rand", type=float, default=1.)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[4096, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 4096])
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--device", type=int)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true',default=False)

    args = parser.parse_args()

    main(args)