#!/usr/bin/env python3

ARCiS = 'ARCiS'

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import logging
import os
from tqdm import trange
from multiprocessing import Process

def x2Ppoints(x, nTpoints):
    Pmin=1e2
    Pmax=1e-7
    Ppoint = np.empty([len(x), nTpoints])
    for j in range(len(x)):
        Ppoint[j, 0] = 10**(np.log10(Pmin) + np.log10(Pmax/Pmin) * (1- x[j]**(1/nTpoints)))
        for i in range(1, nTpoints):
            Ppoint[j,i] = 10**(np.log10(Ppoint[j,i-1]) + np.log10(Pmax/Ppoint[j,i-1]) * (1- x[j]**(1/(nTpoints-i+1))))
    return Ppoint

def simulator(parameters, directory, r, input_file, freeT, nTpoints, n, n_obs, size):
    fname = directory+'/round_'+str(r)+str(n)+'_samples.dat'
    if freeT:
        parametergrid = np.empty([parameters.shape[0], parameters.shape[1]+nTpoints-1])
        parametergrid[:, 0:nTpoints+1] = parameters[:,0:nTpoints+1]
        Ppoints = x2Ppoints(parameters[:,nTpoints+1], nTpoints)
        parametergrid[:,nTpoints+1:nTpoints+1+nTpoints] = np.log10(Ppoints)
        parametergrid[:,nTpoints+1+nTpoints:] = parameters[:, nTpoints+2:]
    else:
        parametergrid = parameters
    np.savetxt(fname, parametergrid)

    print('Running ARCiS')
    print('Computing '+str(len(parameters))+' models for process '+str(n))
    logging.info('Running ARCiS')
    logging.info('Computing '+str(len(parameters))+' models for process '+str(n))
    os.system('cd .. ; '+ARCiS + ' '+input_file + ' -o '+directory+'/round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)

    dirx = directory + '/round_'+str(r)+str(n)+'_out/'
        
    arcis_spec = np.zeros([parameters.shape[0], size])
        
    print('Reading ARCiS output')
    logging.info('Reading ARCiS output')
    
    T = np.zeros([parameters.shape[0], 25])
    P = np.zeros(25)
    
    for i in trange(parameters.shape[0]):
        if i+1<10:
            model_dir = dirx + 'model00000'+str(i+1)
        elif i+1<100:
            model_dir = dirx + 'model0000'+str(i+1)
        elif i+1<1000:
            model_dir = dirx + 'model000'+str(i+1)
        elif i+1<1e4:
            model_dir = dirx + 'model00'+str(i+1)
        try:
            PT = np.loadtxt(model_dir+'/mixingratios.dat')
            T[i] = PT[:,0]
            P = PT[:,1]
            l=[0]
            for j in range(n_obs):
                if j+1<10:
                    obsn = '/obs00'+str(j+1)
                elif j+1<100:
                    obsn = '/obs0'+str(j+1)
                phasej = np.loadtxt(model_dir+obsn)[:,1]
                l.append(len(phasej))
                arcis_spec[i][sum(l[:j+1]):sum(l[:j+2])] = phasej
        except:
            print('Couldn\'t store model ', model_dir)
        
        np.save(directory+'/T_round_'+str(r)+'.npy',T)
        np.save(directory+'/P.npy',P)
    
    return arcis_spec
