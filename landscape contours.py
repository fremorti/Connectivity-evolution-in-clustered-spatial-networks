import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networks import spatialpoint_generator, DM_AMexp
from statsmodels.nonparametric.smoothers_lowess import lowess
start = time.process_time()


#default values
nodes = 9
reps = 10
crad = 0.75
scale = 0.1

data = []
def dfgenerator(xname, xs, yname = '', ys = [1], reps = reps):
    """generate a dataframe that includes distance-based network and cost-based network metrics for networks that vary
    for a generating parameter (xname for values xs) and possibly a second generating parameter (yname for values ys)
    replacated a number of times (reps)"""
    p = {'scale': scale, 'crad': crad, 'cluster': cluster, '': []} #these are the three generating parameters to vary: scale = survival decay parameter cd; crad = clustering factor and cluster = population distribution per cluster, '' handles the case with no secong parameter specified
    for y in ys:
        p[yname] = y
        for x in xs:
            p[xname] = x
            for rep in range(reps):
                pos, DM = spatialpoint_generator(nodes, p['cluster'], crad=p['crad'])
                #adjacency matrix calculations
                adj = DM_AMexp(DM, p['scale'])
                np.fill_diagonal(adj, np.nan)
                adj_ = adj[~np.isnan(adj)].reshape(adj.shape[0], adj.shape[1] - 1)
                conn = np.mean(adj_)
                skew = np.var(np.mean(adj_,1))
                clusterdness = np.mean(np.var(adj_, 1))

                #distance matrix calculations
                np.fill_diagonal(DM, np.nan)
                DM_ = DM[~np.isnan(DM)].reshape(DM.shape[0], DM.shape[1] - 1)
                Dconn = np.mean(DM_)
                Dskew = np.var(np.mean(DM_,1))
                Dclusterdness = np.mean(np.var(DM_, 1))
                data.append([p['scale'], p['crad'], p['cluster'], rep, conn, skew, clusterdness, Dconn, Dskew, Dclusterdness])
        print(f'{yname} = {y}')
    return pd.DataFrame.from_records(data,
                                   columns=['scale', 'crad', 'cluster', 'rep', 'cost-based connectivity', 'cost-based skew', 'cost-based modularity',
                                            'isolation (distance-based)', 'distance-based skew', 'distance-based modularity'])

print(str(time.process_time()))

def contourplot(x, y, z):
    """plots a formatted contour plot"""
    yo = r'conductivity parameter $\alpha_s$' if y == 'scale' else y
    fig, ax = plt.subplots()
    cnt = ax.tricontourf(df[x], df[y], df[z], cmap="RdBu_r")
    '''cnt = ax.tricontour(df['crad'], df['scale'], df['landscape_clusterdness'], levels=None, linewidth = 0.1, colors = 'k')'''

    fig.colorbar(cnt)
    ax.set_xlabel(x)
    ax.set_ylabel(yo)
    plt.title(f'{z} in {str(cluster)}')
    if not os.path.exists(f'{os.getcwd()}/figures/landscape contours/'):
        os.makedirs(f'{os.getcwd()}/figures/landscape contours/')
    plt.savefig(f'{os.getcwd()}/figures/landscape contours/{z}_{x}_{y} in {str(cluster)}')
    plt.savefig(f'{os.getcwd()}/figures/landscape contours/{z}_{x}_{y} in {str(cluster)}.pdf')
    #plt.show()
    plt.close()

def xyplot(df, x, y, z = ''):
    """plots a formated scatterplot with lowess trendlines"""
    df[''] = 1
    df[z] = df[z].map(str)
    df[z] = df[z].astype("category")
    cats = df[z].cat.categories
    cs = plt.cm.get_cmap('Dark2', len(cats))
    lines = []

    fig, ax = plt.subplots()
    for n, lvl in enumerate(cats):
        df_ = df[df[z] == lvl]
        ax.plot(df_[x], df_[y], 'o', ms = 2, color = cs(n))
        line, =ax.plot(*np.transpose(lowess(df_[y], df_[x])), '-', ms = 2, color = cs(n))
        lines.append(line)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.title(f'{y}')
    if z:
        ax.legend(lines, list(cats), title = z)
    if not os.path.exists(f'{os.getcwd()}/figures/euclidian distance/'):
        os.makedirs(f'{os.getcwd()}/figures/euclidian distance/')
    plt.savefig(f'{os.getcwd()}/figures/euclidian distance/{str(z if z else cluster)}_{x}_{y} full')
    plt.savefig(f'{os.getcwd()}/figures/euclidian distance/{str(z if z else cluster)}_{x}_{y} full.pdf')
    #plt.show()
    plt.close()

#the start end and interval steps for the crad (c) and scale (s) used
cstart, cend, cstep = 0.5, 1, 0.01
sstart, send, sstep = 0.05, 0.85, 0.05

clusters = [[3,3,3], [5,3,1]]
crads = np.around(np.arange(cstart, cend, cstep), 2)
scales = np.around(np.arange(sstart, send, sstep), 2)

#define groups of metrics to plot
metrics = 'scale', 'crad', 'cluster', 'rep', 'cost-based connectivity', 'cost-based skew', 'cost-based modularity',\
          'Isolation (distance-based)', 'distance-based skew', 'distance-based modularity'
landscape_metrics = 'cost-based connectivity', 'cost-based skew', 'cost-based modularity'
geographic_metrics = 'isolation (distance-based)', 'distance-based skew', 'distance-based modularity'

#fig2
df = dfgenerator('crad', crads, 'cluster', clusters)
for y in geographic_metrics: xyplot(df, 'crad', y, 'cluster')

#fig3
df = dfgenerator('crad', crads, 'scale', scales)
for z in landscape_metrics: contourplot('distance-based modularity', 'scale', z)
