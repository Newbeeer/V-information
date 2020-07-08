import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from util_ import *
from entropy_model import *
from PIL import Image
from tqdm import tqdm
from data import *
import  matplotlib.pyplot as plt
from model_cifar import ResNet18

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--device', type=int,
                    default='cifar', help='device number')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=2,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=16,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument("--f1", type=int)
parser.add_argument("--f2", type=int)
parser.add_argument('--idx', type=int)
parser.add_argument('--sa', type=float, default=2,
                    help='sa level')
args = parser.parse_args()

# reproducibility

np.random.seed(args.seed)
torch.cuda.set_device(args.device)

sample_batch_size = 25
obs = (1, 64, 64)
input_channels = obs[0]
rescaling = lambda x: (x - .5) * 2.
rescaling_inv = lambda x: .5 * x + .5
kwargs = {'num_workers': 10, 'pin_memory': True, 'drop_last': True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])



loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake)
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
        minimum = 1000000
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
            mat[i,j] = 1
    return mat


def loss_fn(recon_x, x):
    l2 = ((recon_x - x) ** 2).sum()
    return (l2) / x.size(0)

print('starting training')

train_loss = 0.
global_perturb_level = args.sa


def Condition_H(model,pnn,pnn_optimizer):

    for epoch in range(10):

        torch.cuda.synchronize()
        train_loss = 0.0
        time_ = time.time()
        cnt = 0.0
        for batch_idx, (x, target) in enumerate(tqdm(train_data_loader_cifar)):

            img = x[0].squeeze().permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave('./image/ori.png', img)
            # Note:plt doesn't support tensor
            for i in range(2,200):
                Sa_trans = SaturationTrasform(i)
                y = Sa_trans(x[0])
                img = y.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                plt.imsave('./image/sa-'+str(i)+'.png', img)
            break

        break


Condition_H(0,0,0)


#np.save("frames_14.npy",a)

'''
node_num = 5
g = adj2dict(mat)

min_tree = 1000000
tree_dict = dict()

for i in range(node_num):

    h = mst(int(i), g)
    val = 0.0
    s_ = []
    flag = False
    for s in h:
        s_ += [s]
        for k in h[s]:
            val += mat[s, k]
            if k in s_:
                flag = True
    if flag:
        continue

    if val < (min_tree - 0.01):
        min_tree = val
        tree_dict[0] = h
        print(i)

print(tree_dict[0])
T = dict2adj(tree_dict[0], node_num)
print(T)
'''


