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

def simulator(parameters, directory, r, input_file, n, n_obs):
    fname = directory+'round_'+str(r)+str(n)+'_samples.dat'
    np.savetxt(fname, parameters)

    print('Running ARCiS')
    print('Computing '+str(len(parameters))+' models for process '+str(n))
    logging.info('Running ARCiS')
    os.system('cd .. ; '+ARCiS + ' '+input_file + ' -o '+directory+'round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)

    dirx = directory + 'round_'+str(r)+str(n)+'_out/'

    size = 0
    for j in range(n_obs):
        if j+1<10:
            obsn = 'obs00'+str(j+1)
        elif j+1<100:
            obsn = 'obs0'+str(j+1)
        phasej = np.loadtxt(dirx+'model000001/'+obsn)[:,1]
        size += len(phasej)
        
    arcis_spec = np.zeros([parameters.shape[0], size])
        
    print('Reading ARCiS output')
    logging.info('Reading ARCiS output')
    
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
    
    return arcis_spec
