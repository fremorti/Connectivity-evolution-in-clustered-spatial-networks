
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import math
from scipy.spatial import distance_matrix
import random as rnd
import os



def MDS_positions(DM):
    """return the 2D positions of a multi dimensional scaling of a distance-matrix. Used on non-spatially explicit networks"""
    mds_model = manifold.MDS(n_components = 2, dissimilarity = 'precomputed')
    mds_fit = mds_model.fit(DM)
    return mds_fit.embedding_

def DM_generator(nodes, cf, clusters = None):
    """generate spatially non-explicit clusters
        cf: clustering factor: distance of patches in different clusters compared to within clusters"""
    #(kw)arg handling
    if not(clusters) :
        clusters = np.array([nodes])
    elif type(clusters) == int:
        div = nodes//clusters
        mod = nodes%clusters
        clusters = np.array([div+1]*mod+[div]*(clusters-mod))
    else:
        assert sum(clusters) == nodes, 'total of cluster size does not correspond with total number of nodes'

    a = np.ones((nodes, nodes))*cf #afstand tussen nodes van verschillende clusters; potentieel random
    i = 0
    for n in clusters:
        a[i:i+n, i:i+n] = np.ones((n, n)) #afstand tussen nodes van zelfde clusters; potentieel random
        i += n
    np.fill_diagonal(a, 0)
    b = a/np.mean(a)
    return MDS_positions(b),b



def spatialpoint_generator(nodes, clusters = None, lim = 1, crad = 0):
    """generate spatially explicit networks from a 2D spatial pattern
        cf: clustering factor: distance of patches in different clusters compared to within clusters"""
    #(kw)arg handling
    if clusters is None :
        clusters = np.array([nodes])
    elif type(clusters) == int:
        div = nodes//clusters
        mod = nodes%clusters
        clusters = np.array([div+1]*mod+[div]*(clusters-mod))
    else:
        assert sum(clusters) == nodes, 'total of cluster size does not correspond with total number of nodes'

    centroids = np.array([(crad*math.cos(a), crad*math.sin(a)) for a in np.arange(0, 2*math.pi, 2*math.pi/len(clusters))])
    coords = []
    crad_ = lim-crad
    for i, n in enumerate(clusters):
        for node in range(n):
            a, r = rnd.uniform(0, math.pi), crad_*rnd.uniform(-1, 1)
            coords+=[(centroids[i, 0]+r*math.cos(a), centroids[i, 1]+r*math.sin(a))]

    return np.array(coords), distance_matrix(coords, coords)

def DM_AM(DM, unitsurv):
    '''distance to survival probability conversion: proportional decrease of survival probability with distance
    unitsurv: survival probability at distance = 1'''
    AM = 1/((1/unitsurv-1)*DM+1)
    np.fill_diagonal(AM, 0)
    return AM

def DM_AMlin(DM, unitsurv):
    '''distance to survival probability conversion: linear decrease of survival probability with distance
    unitsurv: survival probability at distance = 1'''
    AM = 1-(1-unitsurv)*DM
    AM[AM<0] = 0
    np.fill_diagonal(AM, 0)
    return AM

def DM_AMexp(DM, unitsurv):
    '''distance to survival probability conversion: exponential decrease of survival probability with distance
    unitsurv: survival probability at distance = 1'''
    AM = np.exp(np.log(unitsurv)*DM)
    np.fill_diagonal(AM, 0)
    return AM

def AM_DMexp(AM, unitsurv):
    """survival probability to distance conversion: exponential decrease of survival probability with distance
    unitsurv: survival probability at distance = 1"""
    np.fill_diagonal(AM, 1)
    return np.log(AM)/np.log(unitsurv)

def total_cost_clustering(uslist, nodes, clusters, crads, rep = 10):
    totalcosts = np.array([[[np.sum(DM_AM(spatialpoint_generator(nodes, clusters, crad = crad)[1], us)) for _ in range(rep)] for
                   us in uslist] for
                  crad in crads])
    meantc, stdtc = np.mean(totalcosts, axis = 2), np.std(totalcosts, axis = 2)

    totaldists = np.array([[np.sum(spatialpoint_generator(nodes, clusters, crad = crad)[1]) for _ in range(rep)] for
                   crad in crads])
    meantd, stdtd = np.mean(totaldists, axis=1), np.std(totaldists, axis=1)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(crads, meantd)
    axs[0].fill_between(crads, meantd-stdtd, meantd+stdtd, alpha = 0.2)
    axs[0].set_ylabel('total network distance')
    axs[1].plot(crads, meantc)
    for n, us in enumerate(uslist):
        axs[1].fill_between(crads, meantc[:, n]-stdtc[:, n], meantc[:, n]+stdtc[:, n], alpha = 0.2)
    axs[1].set_ylabel('total network costs')
    axs[1].legend([str(us) for us in uslist])
    plt.show()


def mean_weight_var(G, weight = 'weight'):
    """the mean of edge-weight variance per node,
    how variably connected is an average node to the others --> modularity in the main text, clustering"""
    allweights = [[e for ncol, e in enumerate(row) if ncol != nrow] for nrow, row in enumerate(nx.to_numpy_array(G))]
    return np.mean(np.var(allweights, axis = 1))

def degree_mean(G, weight = 'weight'):
    """mean degree among nodes
    overall isolation for distance matrices, connectivity for cost- or realized dispersal matrices"""
    allweights = [e for nrow, row in enumerate(nx.to_numpy_array(G)) for ncol, e in enumerate(row) if ncol > nrow]
    return np.mean(allweights)

def degree_var(G, weight = 'weight'):
    """variance in mean degree among nodes
    variance among nodes in how well connected to the other nodes --> network skew in the main text, related to centrality"""
    alldegrees = [G.degree(n, weight = 'weight') for n in G]
    return (np.var(alldegrees))



'''
#fig S1-2
for layout in ([5,3,1], [3,3,3]):
    for crad in np.arange(0.5, 1, 0.05):
        pos, DM = spatialpoint_generator(9, layout, crad = crad)
        G = nx.from_numpy_array(DM)
        fig, ax = plt.subplots()
        nx.draw(G, pos = pos, with_labels = 1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        plt.title(f'{layout}, crad: {crad:.2f}', size = 28)
        if not os.path.exists(f'{os.getcwd()}/figures/networks/'):
            os.makedirs(f'{os.getcwd()}/figures/networks/')
        plt.savefig(f'{os.getcwd()}/figures/networks/{layout}_{crad:.2f}.png')
        plt.savefig(f'{os.getcwd()}/figures/networks/{layout}_{crad:.2f}.pdf')
        plt.close()
'''