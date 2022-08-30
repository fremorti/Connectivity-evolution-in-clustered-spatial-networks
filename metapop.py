"""
Metapop IBM in generated spatial networks
author: Frederik Mortier
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
import pandas as pd
from networks import spatialpoint_generator, degree_mean, degree_var, mean_weight_var, DM_AMexp
import copy
import os
from statsmodels.nonparametric.smoothers_lowess import lowess


class Individual:
    """Class that regulates individuals and their properties: location (node) and dispersal propensity (disp)"""

    def __init__(self, disp, node):
        """Initialization"""
        self.disp = disp
        self.node = node  # populatie nr als controle


class Metapopulation:
    """Contains the whole population, regulates daily affairs"""

    def __init__(self, networksize, K, sigma, tsigma, maxR, b, mutationrate, distance, scale, sbias, pos, evo, dnonevo=None):
        """Initialization"""
        self.nnodes = networksize
        self.K = K  # mean K
        self.Ksigma = K*sigma  # stddev K: spatial variance
        self.Ktsigma = K*tsigma  # stddev K: temporal variance
        self.maxR = maxR #population maximum growth rate
        self.b = b #Hassel-Commins population type parameter: 1 = contest competition, very high value = scramble competition
        self.mutationrate = mutationrate #mutation rate of the evolving trait
        self.adj = DM_AMexp(distance, scale) #adjacency matrix of the cost-based network
        self.pos = pos #spatial coordinates of the network
        self.distance = distance #distance matrix of the network
        self.sbias = sbias   # Boolean indicating whether dispersal settlement is biased/distance dependent
        self.evo = evo  # Boolean indicating whether dispersal evolves
        self.dnonevo = dnonevo  # dispersal trait in case of no evolution
        self.initialize_metapop()

    def initialize_metapop(self):
        """Initialize patches, how many individuals and where based on mean K and the spatial variation"""
        self.metaK = np.array([rnd.gauss(self.K, self.Ksigma) for _ in range(self.nnodes)])
        self.metapop = [[Individual(
            rnd.random() if self.evo == 1 else np.random.choice(self.dnonevo) if type(
                self.dnonevo) == list else self.dnonevo, n) for _ in range(int(csize))] for n, csize in
                        enumerate(self.metaK)]

    def metapopdynamics(self, data, time):
        """one time step (generation) of the metapopulation"""
        self.metapopold = copy.deepcopy(self.metapop)
        self.metapopoff = [[] for _ in range(self.nnodes)]
        self.metapop = [[] for _ in range(self.nnodes)]
        #spatio-temporal variation in K
        self.metatK = [max(0.01, K + rnd.gauss(0, self.Ktsigma)) for K in self.metaK]

        #reproduction and population regulation with a population reaction norm
        for node in range(self.nnodes):
            rnd.shuffle(self.metapopold[node])
            for ind in self.metapopold[node]:
                '''Hassel Commins, cfr. Bonte et al. 2008-2010 Oikos BMC'''
                a = (self.maxR ** (1 / self.b) - 1) / self.metatK[node]
                nr_offspring = self.maxR * (1 / (1 + a * len(self.metapopold[node])) ** self.b)
                #nr_offspring = self.maxR * (1 / (1 + a * data[node] ** self.b))
                Fitness = np.random.poisson(nr_offspring)

                for offspring in range(Fitness):
                    odisp = rnd.random() if (self.evo == 1 and rnd.random() < self.mutationrate) or (
                        not (self.evo or self.dnonevo)) else ind.disp
                    odisp = np.random.choice(self.dnonevo) if type(self.dnonevo) == list else odisp
                    self.metapopoff[node].append(Individual(odisp, node))

        # disperal
        self.metadisp = np.zeros((self.nnodes, self.nnodes))
        self.ndisp = 0

        for node in range(self.nnodes):
            rnd.shuffle(self.metapopoff[node])
            for ind in self.metapopoff[node]:
                if rnd.random() < (ind.disp if self.evo != 'landscape' else 0.05 * (
                        np.sum(self.adj[node]) * self.nnodes / np.sum(self.adj))):
                    targetpatch = np.random.choice(self.nnodes, p=self.adj[node] / np.sum(self.adj[node]) if np.sum(self.adj[node]) and self.sbias else None)  # destination according to adjacency
                    self.ndisp += 1
                    if self.evo == 'landscape' or rnd.random() < self.adj[node, targetpatch]:
                        self.metadisp[node, targetpatch] += 1
                        self.metapop[targetpatch].append(ind)

                else:
                    self.metapop[node].append(ind)


# noinspection PyAttributeOutsideInit,SpellCheckingInspection,PyShadowingNames
class Datacollector:
    """class for data extraction, processing and visualization"""

    def __init__(self, maxtime, meta, clus):
        self.maxtime = maxtime
        self.meta = meta
        self.DM = meta.distance
        self.D = nx.from_numpy_array(meta.adj)
        nodes = self.meta.nnodes
        self.sizeintime = np.zeros((maxtime, nodes))
        self.connectivityintime = np.zeros((maxtime, nodes, nodes))
        self.dispintime = []
        self.disprateintime = np.zeros(maxtime)
        self.metasizeintime = np.zeros(maxtime)
        self.patchextinction = np.zeros(maxtime)
        self.clus = clus
        self.clist = []
        s = 0
        for c in self.clus:
            self.clist.append(range(s, s + c))
            s += c

    def collecttimestep(self, time):
        """information extracted each generation and added to time-series dfs"""
        self.sizeintime[time] = np.array([len(local) for local in self.meta.metapop])
        self.metasizeintime[time] = sum(self.sizeintime[time])
        self.connectivityintime[time] = self.meta.metadisp
        self.dispintime.append([[ind.disp for ind in local] for local in self.meta.metapop])
        self.disprateintime[time] = self.meta.ndisp / np.sum([len(local) for local in self.meta.metapopoff])
        self.patchextinction[time] = np.count_nonzero(not self.meta.metapop) / self.meta.nnodes

    #The following are functions that calculate metrics for network time-series
    def meanconn(self, generations):
        """the mean of local dispersers between two nodes over a number of generations"""
        return np.mean(self.connectivityintime[-generations:], 0)

    def meandisp(self, generations):
        """the mean dispersal trait over a number of generations"""
        return np.nanmean([[np.mean(pop) for pop in gen] for gen in self.dispintime[-generations:]], 0)

    def meanclusterdisp(self, generations):
        return [np.mean([x for gen in self.dispintime[-generations:] for pop in c for x in gen[pop]])
                if ([x for gen in self.dispintime[-generations:] for pop in c for x in gen[ pop]]) else 0
                for c in self.clist]

    def clustersize(self, generations):
        return np.array([[np.sum([gen[pop] for pop in c]) for c in self.clist] for gen in self.sizeintime[-generations:]])

    def clustermean(self, generations):
        return np.array([[np.mean([gen[pop] for pop in c]) for c in self.clist] for gen in self.sizeintime[-generations:]])

    def patchsd(self, generations):
        return np.std(self.sizeintime[-generations:], axis=0)

    def meanlocal(self, generations):
        """the mean local population size over a number of generations"""
        return np.mean(self.sizeintime[-generations:], 0)

    def meanmeta(self, generations):
        return np.mean(self.metasizeintime[-generations:])

    def meanext(self, generations):
        return np.mean(self.patchextinction[-generations:])

    def popvariabilities(self, generations):
        meanmeta = self.meanmeta(generations)
        alpha = (np.sum(self.patchSD) / meanmeta) ** 2

        SDmetapopsize = np.std(self.metasizeintime[-generations:])
        gamma = (SDmetapopsize / meanmeta) ** 2

        beta = alpha - gamma
        beta_ = alpha / gamma

        return alpha, gamma, beta, beta_

    def amongclustervariabilities(self, generations):
        meanmeta = self.meanmeta(generations)
        calpha = (np.sum(self.clusSD) / meanmeta)**2

        cbeta = calpha - self.gamma
        cbeta_ = calpha / self.gamma

        return calpha, cbeta, cbeta_

    def withinclustervariabilities(self, generations):
        meanc = self.csize
        walphasum = np.array([np.sum([self.patchSD[pop] for pop in c]) for c in self.clist])
        withinalpha = (walphasum/meanc)**2

        percgamma = (self.clusSD/self.csize)**2

        withinbeta = withinalpha - percgamma
        withinbeta_ = withinalpha / percgamma

        return withinalpha, percgamma, withinbeta, withinbeta_


    def generate_runmetrics(self, generations):
        #the collected metrics for one model run, so from a time-series
        # landscape metrics
        self.landscape_connectedness = degree_mean(self.D)
        self.landscape_vardegree = degree_var(self.D)
        self.landscape_weightvar = mean_weight_var(self.D)

        #geographoc metrics
        self.geographic_connectedness = np.mean(self.DM)
        self.geographic_skew = np.var(np.mean(self.DM,1))
        self.geographic_clusterdness = np.mean(np.var(self.DM, 1))

        # metapop metrics
        self.patchSD = self.patchsd(generations)
        self.alpha, self.gamma, self.beta, self.beta_ = self.popvariabilities(generations)
        self.metasize = self.meanmeta(generations)
        self.mean_disp = np.mean(self.meandisp(generations))
        self.mean_conn = np.mean(self.meanconn(generations))

        # connectivity network metrics
        self.CM = self.meanconn(generations)
        self.C = nx.from_numpy_array(self.CM, create_using=nx.DiGraph)
        self.vardegree = degree_var(self.C, 'weight')
        self.weightvar = mean_weight_var(self.C, 'weight')

        # local differenes metrics
        self.localdisp = self.meandisp(generations)
        self.var_disp = np.var(self.localdisp)

        # cluster stats
        self.cdisp = self.meanclusterdisp(generations)
        self.var_disp_cluster = np.var(self.cdisp)
        self.csizet = self.clustersize(generations)
        self.csize = np.mean(self.csizet, axis = 0)
        self.cmean = np.mean(self.clustermean(generations), axis = 0)

        self.clusSD = np.std(self.csizet, axis = 0)
        self.calpha, self.cbeta, self.cbeta_ = self.amongclustervariabilities(generations)
        self.withinalpha, self.withingamma, self.withinbeta, self.withinbeta_ = self.withinclustervariabilities(generations)

        intracluster_disp = [np.sum(self.CM[c[0]:c[-1]+1, c[0]:c[-1]+1]) for c in self.clist]
        self.interclusterdispersalratio = 1-(sum(intracluster_disp)/np.sum(self.CM))
        #components of the dispersal network
        self.CMthresh = self.CM
        self.CMthresh[self.CMthresh < (0.2 / self.meta.K)] = 0
        self.Cthresh = nx.from_numpy_array(self.CMthresh, create_using=nx.DiGraph)
        self.isolates = nx.number_of_isolates(self.Cthresh)
        self.components = nx.number_weakly_connected_components(self.Cthresh)
        self.biggest_component = np.max([len(x) for x in nx.weakly_connected_components(self.Cthresh)])


def compareplot(data, x, ys, title="", folder=None):
    """plot a per-cluster metric in relation to another metric, for each cluster"""
    ys = [ys] if type(ys) == int else ys
    cs = plt.cm.get_cmap('Set1', len(ys))

    fig, ax = plt.subplots()
    lines = []
    for n, y in enumerate(ys):
        ax.scatter(data[x], data[y], s=5, color=cs(n))
        line, = ax.plot(*np.transpose(lowess(data[y], data[x], frac=0.75)), color=cs(n))
        lines.append(line)
    ax.legend(lines, ys)
    plt.xlabel(f'{x}')
    plt.ylabel(title)
    plt.title(title)
    if not os.path.exists(f'{os.getcwd()}/figures/{x}/{NODES}_{folder if folder else CLUSTERS}/'):
        os.makedirs(f'{os.getcwd()}/figures/{x}/{NODES}_{folder if folder else CLUSTERS}/')
    plt.savefig(f'{os.getcwd()}/figures/{x}/{NODES}_{folder if folder else CLUSTERS}/{title}.png')
    plt.savefig(f'{os.getcwd()}/figures/{x}/{NODES}_{folder if folder else CLUSTERS}/{title}.pdf')
    plt.close()


def evoplot(data, x, y, var, name):
    """plot a metric as a function of another metric"""
    cats = data['bound'].astype('category').cat.categories
    data_evo = data[data['bound'] == cats[0]]
    cs = plt.cm.get_cmap('Set1', len(cats))
    fig, ax = plt.subplots()
    ax.scatter(data_evo[x], data_evo[y], s=5, color=cs(0))

    line1, = ax.plot(*np.transpose(lowess(data_evo[y], data_evo[x], frac=0.75)), color=cs(0))
    lines = [line1]
    for ncat in range(1, len(cats)):
        data_u = data[data['bound'] == cats[ncat]]
        ax.scatter(data_u[x], data_u[y], s=5, color=cs(ncat))
        line, = ax.plot(*np.transpose(lowess(data_u[y], data_u[x])), color=cs(ncat))
        lines.append(line)

    ax.legend(lines, list(cats), title = r'$c_d$' if var == 'scale' else var)
    labelname = {'crad': 'crad', 'rep': 'repetition', 'bound':'bound',
                  'alpha': r'$\alpha$ variability', 'gamma': r'$\gamma$ variability', 'beta': r'$\beta_2$ variability', 'beta_': r'$\beta$ variability', 'metapopsize': 'metapopulation size',
                  'cluster_alpha': r'between cluster $\alpha$ variability', 'cluster_beta': r'between cluster $\beta_2$ variability',
                  'cluster_beta_': r'between cluster $\beta$ variability', 'alpha biggest cluster': r'$\alpha$ variability of biggest cluster',
                  'beta_ biggest cluster': r'$\beta$ variability of biggest cluster', 'gamma biggest cluster': r'$\gamma$ variability of biggest cluster',
                  'mean_disp_trait': 'mean dispersal trait',
                  'disp_var': 'dispersal trait variance', 'disp_cluster_var': 'between cluster dispersal trait variance',
                  'beta_disp': 'dispersal trait variance', 'beta_cluster_disp': 'between cluster dispersal trait variance',
                  'realized_connectivity': 'realized connectivity', 'intercluster_disp_ratio': 'intercluster dispersal ratio',
                  'isolates': 'isolates', 'number_components': 'number of components', 'biggest component': 'biggest component',
                  'dispersal_skew': 'realized dispersal skew', 'dispersal_modularity': 'realized dispersal modularity',
                  'cost-based_connectedness': 'cost-based connectedness', 'cost-based_skew': 'cost-based skew',
                  'cost-based_modularity': 'cost-based modularity', 'distance_connectedness': 'isolation (distance-based)',
                  'distance_skew': 'distance-based skew', 'distance_modularity': 'distance-based modularity',
                 '\u0394_connectivity': '\u0394 connectivity', '\u0394_skew': '\u0394 skew', '\u0394_clusterdness': '\u0394 modularity', '\u0394_modularity': '\u0394 modularity'}
    plt.xlabel(f'{labelname[x]}')
    plt.ylabel(f'{labelname[y]}')
    plt.title(f'{labelname[y]}')

    if not os.path.exists(f'{os.getcwd()}/figures/{x}/{NODES}_{name}/'):
        os.makedirs(f'{os.getcwd()}/figures/{x}/{NODES}_{name}/')
    plt.savefig(f'{os.getcwd()}/figures/{x}/{NODES}_{name}/{y}.png')
    plt.savefig(f'{os.getcwd()}/figures/{x}/{NODES}_{name}/{y}.pdf')
    plt.close()


def run(nodes, clus, maxtime, K, r, b, sigma, tsigma, mut, crad, pos, DM, scale, sbias, evo=1, dnonevo=None):
    meta = Metapopulation(nodes, K, sigma, tsigma, r, b, mut, DM, scale, sbias, pos, evo, dnonevo)
    data = Datacollector(maxtime, meta, clus)
    for time in range(maxtime):
        meta.metapopdynamics(data.sizeintime[time-delay] if time>=delay else data.sizeintime[time-delay]+K, time)
        data.collecttimestep(time)

    data.generate_runmetrics(generations)
    return data


def runs(nodes, clus, maxtime, K, r, b, sigma, tsigma, mut, scale, sbias, reps=10,
         cstart=0.5, cend=1, cstep=0.05, comparisons=None):
    """function that generates multiple model runs of the IBM for set parameters, extracts metrics and plots them"""
    #handles comparisons of the evolutionary model 'evo', with different scenarios of fixed dispersal
    comparisons = [] if comparisons is None else comparisons
    supported = set(('upper', 'lower', 'mean', 'landscape', 'meanvar'))

    crads = np.arange(0, cend - cstart + cstep, cstep) + cstart
    datas = []
    localdisp = []
    cdisp = []
    networks = [[[] for _ in range(reps)] for _ in crads]
    for cn, crad in enumerate(crads):
        for rep in range(reps):
            pos, DM = spatialpoint_generator(nodes, clus, crad=crad)
            data = run(nodes, clus, maxtime, K, r, b, sigma, tsigma, mut, crad, pos, DM, scale, sbias, 1)
            datas.append((crad, rep, 'evo',
                          data.alpha, data.gamma, data.beta, data.beta_, data.metasize, data.calpha, data.cbeta, data.cbeta_,
                          data.mean_disp, data.var_disp,
                          data.var_disp_cluster, data.mean_conn, data.interclusterdispersalratio,
                          data.isolates, data.components, data.biggest_component, data.vardegree, data.weightvar,
                          data.landscape_connectedness, data.landscape_vardegree, data.landscape_weightvar,
                          data.geographic_connectedness, data.geographic_skew, data.geographic_clusterdness))
            #store the network from this run
            networks[cn][rep] = (pos, DM, data.mean_disp, [x for gen in data.dispintime[-3:] for pop in gen for x in
                                                           pop] if 'meanvar' in comparisons else None)
            localdisp.append([crad, rep, 'evo', data.landscape_weightvar] + list(data.localdisp))
            cdisp.append([crad, rep, 'evo', data.landscape_weightvar] + list(data.cdisp)+ list(data.csize)+ list(data.withinalpha))

        print(f'cluster factor {crad}')
    alld = [rep[2] for cn in networks for rep in cn]


    fixed_disp = set(comparisons).intersection(supported)
    if fixed_disp:
        for n, bound in enumerate(fixed_disp):
            for cn, crad in enumerate(crads):
                for rep in range(reps):
                    #recall the networks from the 'evo' runs
                    pos, DM, d, ddis = networks[cn][rep]
                    #dispersal of the individuals are set to the minimum (lower), maximum (upper) or mean (mean and landscape), or resampled from the dispersal propensity in the last three geenerations, 'landscape'-scenarios eliminate any dispersal cost
                    data = run(nodes, clus, maxtime, K, r, b, sigma, tsigma, mut, crad, pos, DM, scale, sbias,
                               'landscape' if bound == 'landscape' else 0, dnonevo=
                               {'lower': min(alld), 'upper': max(alld), 'mean': d, 'random': None, 'landscape': d,
                                'meanvar': ddis}[bound])
                    datas.append((crad, rep, bound,
                                  data.alpha, data.gamma, data.beta, data.beta_, data.metasize, data.calpha, data.cbeta, data.cbeta_,
                                  data.mean_disp,
                                  data.var_disp, data.var_disp_cluster, data.mean_conn, data.interclusterdispersalratio,
                                  data.isolates, data.components, data.biggest_component, data.vardegree,
                                  data.weightvar,
                                  data.landscape_connectedness, data.landscape_vardegree, data.landscape_weightvar,
                                  data.geographic_connectedness, data.geographic_skew, data.geographic_clusterdness))
                    localdisp.append([crad, rep, bound, data.landscape_weightvar] + list(data.localdisp))
                    cdisp.append([crad, rep, bound, data.landscape_weightvar] + list(data.cdisp) + list(data.csize)+ list(data.withinalpha))
                print(f'cluster factor {crad}')
    #generate a df from all network metrics from all runs
    df = pd.DataFrame.from_records(datas, columns=('crad', 'rep', 'bound',
                                                   'alpha', 'gamma', 'beta', 'beta_', 'metapopsize', 'cluster_alpha', 'cluster_beta', 'cluster_beta_', 'mean_disp_trait',
                                                   'disp_var', 'disp_cluster_var', 'realized_connectivity', 'intercluster_disp_ratio',
                                                   'isolates', 'number_components', 'biggest component',
                                                   'dispersal_skew', 'dispersal_modularity',
                                                   'cost-based_connectedness', 'cost-based_skew',
                                                   'cost-based_modularity', 'distance_connectedness',
                                                   'distance_skew', 'distance_modularity'))
    df['\u0394_connectivity'] = df['realized_connectivity'] / df['cost-based_connectedness']
    df['\u0394_skew'] = df['dispersal_skew'] / df['cost-based_skew']
    df['\u0394_modularity'] = df['dispersal_modularity'] / df['cost-based_modularity']

    #generata a df with per-cluster metrics
    dfcdisp = pd.DataFrame.from_records(cdisp,
                                        columns=['crad', 'rep', 'bound', 'cost-based modularity'] + [
                                            f'c{n} ({x})' for n, x in enumerate(clus)] + [f'size_c{n} ({x})' for n, x in enumerate(clus)]+
                                                [f'alpha_c{n} ({x})' for n, x in enumerate(clus)])

    if not os.path.exists(f'{os.getcwd()}/dataframes/'):
        os.makedirs(f'{os.getcwd()}/dataframes/')
    df.to_csv(f'{os.getcwd()}/dataframes/data{comparisons}_{nodes}_{clus}_{sbias}.csv')

    #plots all metrics in function of a desired metric
    for y in df[df.columns[3:]]:
        evoplot(df, 'cost-based_modularity', y, clus, f'{clus}_{sbias}')
        evoplot(df, 'crad', y, clus, f'{clus}_{sbias}')

    #compares the per-cluster metrics
    compareplot(dfcdisp[dfcdisp['bound'] == 'evo'], 'cost-based modularity',
                [f'c{n} ({x})' for n, x in enumerate(clus)], title='dispersal trait per cluster', folder = f'{clus}_{sbias}')
    compareplot(dfcdisp[dfcdisp['bound'] == 'evo'], 'crad',
                [f'c{n} ({x})' for n, x in enumerate(clus)], title='dispersal trait per cluster', folder = f'{clus}_{sbias}')
    compareplot(dfcdisp[dfcdisp['bound'] == 'evo'], 'cost-based modularity',
                [f'size_c{n} ({x})' for n, x in enumerate(clus)], title='mean population size per cluster', folder = f'{clus}_{sbias}')
    compareplot(dfcdisp[dfcdisp['bound'] == 'evo'], 'crad',
                [f'size_c{n} ({x})' for n, x in enumerate(clus)], title='mean population size per cluster', folder = f'{clus}_{sbias}')


def plot_from_df(df, clus, iters, sbias):
    '''re-plot from saved dataframes'''
    for y in df[df.columns[-3:]]:
        evoplot(df, 'mean_disp_trait', y, clus, f'compare {clus}_{len(iters)}_{sbias}')
    evoplot(df, 'mean_disp_trait', 'alpha', clus, f'compare {clus}_{len(iters)}_{sbias}')



def compare_clus_runs(var, iters, nodes, clus, maxtime, K, r, b, sigma, tsigma, mut, scale, sbias, reps=10,
                      cstart=0.5, cend=1, cstep=0.05):
    #generates and compares runs over a range for a given key-word parameter
    crads = np.arange(0, cend - cstart + cstep, cstep) + cstart
    datas = []

    #key-word parameters
    p = {'nodes' : nodes, 'cluster':clus, 'maxtime':maxtime, 'K':K, 'r':r, 'b':b, 'sigma':sigma, 'tsigma':tsigma,
              'mut':mut, 'scale':scale, 'settlement_bias':sbias}
    for x in iters:
        p[var] = x
        localdisp = []
        cdisp = []
        for cn, crad in enumerate(crads):
            for rep in range(reps):
                pos, DM = spatialpoint_generator(p['nodes'], p['cluster'], crad=crad)
                data = run(p['nodes'], p['cluster'], p['maxtime'], p['K'], p['r'], p['b'], p['sigma'], p['tsigma'], p['mut'], crad, pos, DM, p['scale'], p['settlement_bias'], 1)
                datas.append((crad, rep, str(x),
                              data.alpha, data.gamma, data.beta, data.beta_, data.metasize, data.calpha, data.cbeta, data.cbeta_,
                              data.withinalpha[0], data.withinbeta_[0], data.withingamma[0], data.mean_disp,
                              data.var_disp, data.var_disp_cluster, data.mean_conn, data.interclusterdispersalratio,
                              data.isolates, data.components, data.biggest_component, data.vardegree, data.weightvar,
                              data.landscape_connectedness, data.landscape_vardegree, data.landscape_weightvar,
                              data.geographic_connectedness, data.geographic_skew, data.geographic_clusterdness))
                localdisp.append(
                    [crad, rep, x, data.landscape_weightvar] + list(data.localdisp))
                cdisp.append([crad, rep, x, data.landscape_weightvar] + list(data.cdisp)+ list(data.cmean) + list(data.withinalpha) + list(data.withinbeta_) + list(data.withingamma))

            print(f'cluster factor {crad}')

        #generata a df with per-cluster metrics
        dfcdisp = pd.DataFrame.from_records(cdisp, columns=['crad', 'rep', 'bound', 'cost-based_modularity'] +
                                                           [f'disp_c{n+1} ({x})' for n, x in enumerate(p['cluster'])] +
                                                           [f'size_c{n+1} ({x})' for n, x in enumerate(p['cluster'])] +
                                                           [f'alpha_c{n+1} ({x})' for n, x in enumerate(p['cluster'])] +
                                                           [f'beta_c{n+1} ({x})' for n, x in enumerate(p['cluster'])]+
                                                           [f'gamma_c{n+1} ({x})' for n, x in enumerate(p['cluster'])])
        #plots all metrics in function of a desired metric
        compareplot(dfcdisp, 'crad',
                    [f'disp_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'dispersal trait per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'cost-based_modularity',
                    [f'disp_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'dispersal trait per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'crad',
                    [f'size_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'mean population size per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'cost-based_modularity',
                    [f'size_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'mean population size per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'crad',
                    [f'alpha_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'meanlocal population variability per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'cost-based_modularity',
                    [f'alpha_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'local population variability per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'crad',
                    [f'beta_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'asynchronicity per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'cost-based_modularity',
                    [f'beta_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'asynchronicity per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'crad',
                    [f'gamma_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'total population size variability per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')
        compareplot(dfcdisp, 'cost-based_modularity',
                    [f'gamma_c{n+1} ({x})' for n, x in enumerate(p['cluster'])],
                    title=f'total population size variability per cluster in {var}_{str(x)}', folder=f'compare {var}_{len(iters)}_{sbias}')

        print(f'{var}: {x}')
    #generate a df from all network metrics from all runs
    df = pd.DataFrame.from_records(datas, columns=('crad', 'rep', 'bound',
                                                   'alpha', 'gamma', 'beta', 'beta_', 'metapopsize', 'cluster_alpha', 'cluster_beta', 'cluster_beta_',
                                                   'alpha biggest cluster', 'beta_ biggest cluster', 'gamma biggest cluster', 'mean_disp_trait',
                                                   'disp_var', 'disp_cluster_var', 'realized_connectivity', 'intercluster_disp_ratio',
                                                   'isolates', 'number_components', 'biggest component',
                                                   'dispersal_skew', 'dispersal_modularity',
                                                   'cost-based_connectedness', 'cost-based_skew',
                                                   'cost-based_modularity', 'distance_connectedness',
                                                   'distance_skew', 'distance_modularity'))
    df['\u0394_connectivity'] = df['realized_connectivity'] / df['cost-based_connectedness']
    df['\u0394_skew'] = df['dispersal_skew'] / df['cost-based_skew']
    df['\u0394_modularity'] = df['dispersal_modularity'] / df['cost-based_modularity']

    if not os.path.exists(f'{os.getcwd()}/dataframes/'):
        os.makedirs(f'{os.getcwd()}/dataframes/')
    df.to_csv(f'{os.getcwd()}/dataframes/data_compareclusters_{nodes}_{clus}_{var}_{len(iters)}_{sbias}.csv')
    #plots all metrics in function of a desired metric
    for y in df[df.columns[3:]]:
        evoplot(df, 'crad', y, var, f'compare {var}_{len(iters)}_{sbias}')
        evoplot(df, 'cost-based_modularity', y, var, f'compare {var}_{len(iters)}_{sbias}')


############PARAMETER SETTINGS#################
NODES = 9               #number of populations
CLUSTERS = [5, 3, 1]    #population distribution over clusters
MAXTIME = 300           # equilirbrium is reached after 500gen, default: 300
K = 100                 # local K, default: 100
ssigma = 0              # spatialvar in K (percentages), default:0
tsigma = 0.25           # default: 0.25
maxR = 2                # maximum growth rate, default: 2
b = 1                   # Hassel competition parameter, default:1
mutationrate = 0.01     # default: 0.01
crad = 0.9              # clustering factor, default value but in all examples below this is overwritten
SCALE = 0.1             # spatial scale parameter (survival decay oarameter), the survival probability of dispersing 1 distance unit (half of the landscape), default: 0.1
generations = 50        # generations taken into account for metapopulation variability metrics, default: 50
delay =0                # default = 1
settlement_bias = 1     # whether settlement is biased towards closer populations or equally weighted





#Variable to test and levels to test them for
#fig 4, 5, 6, S2.1-2, S3; change CLUSTERS and settlement_bias in the parameter settings section for alternate runs with even clusters and without biased settlement resp
#var, iters = 'scale', [0.02, 0.1, 0.3, 0.5, 0.8]
#compare_clus_runs(var, iters, NODES, CLUSTERS, MAXTIME, K, maxR, b, ssigma, tsigma, mutationrate, SCALE, settlement_bias, reps = 20, cstep=0.05, cend = 0.95)

#fig7a-c
#var, iters = 'cluster', ([5, 3, 1], [3, 3, 3])
#compare_clus_runs(var, iters, NODES, CLUSTERS, MAXTIME, K, maxR, b, ssigma, tsigma, mutationrate, SCALE, settlement_bias, reps = 20, cstep=0.05, cend = 0.95)

#fig 7d-f
#pos, DM = spatialpoint_generator(NODES, CLUSTERS, crad = crad)
#runs(NODES, CLUSTERS, MAXTIME, K, maxR, b, ssigma, tsigma, mutationrate, SCALE, reps=20, cstep=0.05, cend=0.95, comparisons=['mean', 'meanvar'], sbias = settlement_bias)

#fig S4
#var, iters = 'cluster', ([5, 4, 3, 2, 1])
#compare_clus_runs(var, iters, NODES, CLUSTERS, MAXTIME, K, maxR, b, ssigma, tsigma, mutationrate, SCALE, settlement_bias, reps = 20, cstep=0.05, cend = 0.95)


#plot from previously saved csv
#df = pd.read_csv(f'{os.getcwd()}/dataframes/data_compareclusters_{NODES} nodes_{var}_{len(iters)}_{settlement_bias}.csv', encoding = 'utf-8')
#plot_from_df(df, var, iters ,settlement_bias)