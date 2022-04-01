# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:07:16 2021

@author: 20210543
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ODmodel import OD

def create_plot_network(G,N,opinions,vmin,vmax,figtitle,simstr,savef=False):
# Create network for visualization
    edges=[]
    for i in range(N):
        for j in range(i):
            if G[i,j]==1:
                edges.append((i,j))

    cmap = plt.cm.viridis # plt.cm.hot #
    G2 = nx.Graph()
    G2.add_nodes_from(np.arange(0,N))
    G2.add_edges_from(edges)
    pos = nx.spring_layout(G2,seed=10)    
    
 
    plt.figure()
    edgesdraw = nx.draw_networkx_edges(G2, pos, alpha=0.4)
    nodesdraw = nx.draw_networkx_nodes(G2, pos, node_color=opinions, cmap=cmap, node_size=50, vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)
    plt.title(figtitle)
    if savef:
        plt.savefig('figs_maxnb/' + simstr + figtitle + '.png')

N=100
nrtime=2000

dist_removelink = 0.6
prob_removelink = 0.1 
tries_createlink = 10 
dist_createlink = 0.1 
prob_createlink = 0.1 
#maxnb = 10 
tries_valuechange = 10
rate_valuechange = 0.05
tries_opinionchange = 150 
distcd=1
T=0.1

maxnbs = np.arange(1,16)
nrparamchange=len(maxnbs)

sims=100

OPs=np.zeros((sims,N,nrparamchange))
categories=np.zeros((sims,nrparamchange))


for k in range(nrparamchange):
    maxnb = maxnbs[k]
    for l in range(sims):
        print(k,l)
        simstr='k=' + str(k) + 'l=' + str(l)   
        stubbornness = np.random.rand(N) 
        persuasiveness = np.random.rand(N) 
    
        opinions = np.random.rand(N)*2-1
        values = np.random.rand(N)*2-1
        
        G=np.zeros((N,N))
        for i in range(N):
            for j in range(i):
                if (np.random.uniform(0,1)<0.05 and sum(G[i])<maxnb and sum(G[j])<maxnb):
                    G[i,j]=1
                    G[j,i]=1
        nonb=np.where(sum(G)<0.1)[0]
        for i in range(len(nonb)):
            G[nonb[i],nonb[i]-1] = 1
            G[nonb[i]-1,nonb[i]] = 1
            
        opinions,G,category,dist_opinions = OD(G,N,opinions,values,simstr,nrtime,dist_removelink,prob_removelink,tries_createlink,
            maxnb,dist_createlink,prob_createlink,tries_valuechange,rate_valuechange,
            tries_opinionchange,stubbornness,persuasiveness,distcd,T,create_plot_network)
        OPs[l,:,k]=opinions
        categories[l,k]=category
    
# np.save('out_maxnb_1_15_nrparams_15_OPs',OPs)
# np.save('out_maxnb_1_15_nrparams_15_categories',categories)
# np.save('out_maxnb_1_15_nrparams_15_maxnbs',maxnbs)


