# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:18:08 2022

@author: elsweinans
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance
import copy

def OD(G,N,opinions,values,simstr,nrtime,dist_removelink,prob_removelink,tries_createlink,
       maxnb,dist_createlink,prob_createlink,tries_valuechange,rate_valuechange,
       tries_opinionchange,stubbornness,persuasiveness,distcd,T,create_plot_network):
    
    """ function that performs one opinion dynamics simulation with
    G: initial np matrix with links between agents
    N: number of agents
    values: a value for every agent (between -1 and 1)
    simstr: string to find back the plots
    nrtime: length of simulation
    dist_removelink: maximum distance to maintain a link
    prob_removelink: the probability of removing a link with an opinion that is further than dist_removelink away
    tries_createlink: number of times an agent may try to create a new link
    maxnb: maximum number of niehgbors
    dist_createlink: The maximum opinion distance to create a new link
    prob_createlink: probbility of creating a new link if opinion is within dist_createlink
    tries_valuechange: number of agents that may update their value
    rate_valuechange: rate in which agents update their value towards their neighbors
    tries_opinionchange: number of agents that may update their opinion
    stubbornness: the stubbornness of every agent
    persuasiveness: the persuasiveness of every agent
    distcd: distance for the cognitivie dissonance process to work
    T: temperature (stochasticity in opinion updating procedure)
    create_plot_network: function to plot the network and save the fig
    """
    
    titleplot= 'time = 0' 
    create_plot_network(G, N, opinions, -1, 1, titleplot, simstr, savef=True)
    plt.close('all')
    
    dist_opinions=np.zeros(200) # how many iterations used to say that the system has become stable
    dist_opinions[-1]=1
    k=0
    
    
    while (not all(dist_opinions<0.003) and k < nrtime):
        temp_opinion=copy.deepcopy(opinions)
        # step 1: remove neighbors with opdiff>0.5 
        for ind in range(N):
            opnb = opinions*G[ind] # opinions of neighbors
            myop = opinions[ind]*G[ind] #agent's own opinion at location of neighbors
            distop = abs(myop - opnb) # distance of agent's opinion to it's neighbors
            idx=np.where(distop>dist_removelink)[0] # locations where opinion distance higher than dist_removelink
            for j in range(len(idx)):
                if (np.random.uniform(0,1)<prob_removelink and sum(G[ind])>1 and sum(G[idx[j]])>1):
                    G[ind,idx[j]]=0
                    G[idx[j],ind]=0
                    
            # step 2: make new connections
            for j in range(tries_createlink):
                if sum(G[ind])<maxnb:
                    newnb=np.random.randint(0,N) # possible new neighbor
                    while (newnb == ind):
                        newnb=np.random.randint(0,N) # don't pick yourself
                    if abs(opinions[newnb]-opinions[ind])<dist_createlink and G[ind,newnb]==0 and sum(G[newnb]<maxnb):
                        if np.random.uniform(0,1)<prob_createlink:
                            G[ind, newnb] = 1
                            G[newnb, ind] = 1
        
        # step 3: change values
        for j in range(tries_valuechange):
            ind = np.random.randint(0,N)
            valnb = values*G[ind]
            optval = sum(valnb)/sum(abs(valnb)>0)
            distval = optval - values[ind]
            values[ind] = values[ind] + rate_valuechange * distval
                    
        # step 4: change opinions
        for j in range(tries_opinionchange): #2        
            ind = np.random.randint(0,N)
            if np.random.rand() < 1-stubbornness[ind]: # go into opinion-change procedure dependent on stubbornness
                randnew = np.random.rand(1)*2-1 # random new opinion
                opnb = opinions*G[ind] # opinion of neighbors
                distnb=abs(values-values[ind]) #value distance to neighbors
                valuesigns = (G[ind]*[distnb>distcd])[0]*2 # 2 if distnb > distcd, 0 if distnb < distcd
                Eold = sum(abs(valuesigns-abs(opinions[ind]*G[ind] - opnb))*persuasiveness) 
                Enew = sum(abs(valuesigns-abs(randnew*G[ind] - opnb))*persuasiveness)
                dH = Enew - Eold
                if dH < 0:
                    opinions[ind] = randnew
                elif np.random.rand() < exp(-dH/T):
                    opinions[ind]  = randnew
        
        dist_opinions[:-1] = dist_opinions[1::]
        dist_opinions[-1] = wasserstein_distance(opinions,temp_opinion)
                     
        k += 1
        
    titleplot= 'time = ' + str(k)
    create_plot_network(G, N, opinions, -1, 1, titleplot, simstr, savef=True)
    plt.close('all')
            
    plt.figure()
    my_kde = sns.kdeplot(data=opinions, color='b', bw_adjust=0.9)
    plt.xlim(-1,1)
    line = my_kde.lines[0]
    x, y = line.get_data()
    nrpeaks = len(find_peaks(y,height=max(y)/10,prominence=0.1)[0])
    if nrpeaks==1:
        if np.var(opinions)<0.05:
            categories=0
        else:
            categories=1
    elif nrpeaks==2:
        categories=2
    else:
        categories=3
    plt.close('all')
    return opinions, G, categories, dist_opinions