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

def simulator(parameters, directory, r, input_file, n):
    fname = directory+'round_'+str(r)+str(n)+'_samples.dat'
    np.savetxt(fname, parameters)

    print('Running ARCiS')
    logging.info('Running ARCiS')
    os.system('cd .. ; '+ARCiS + ' '+input_file + ' -o '+directory+'round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)

    dirx = directory + 'round_'+str(r)+str(n)+'_out/'
    phaseR = os.path.isfile(dirx+'model000001/phasecurve')
    
    sizet = np.loadtxt(dirx+'model000001/trans').shape[0]
    trans = np.zeros([parameters.shape[0], sizet])
    
    if phaseR:
        sizep = np.loadtxt(dirx+'model000001/phase').shape
        
        ###############################################################
        phase = np.zeros([parameters.shape[0], sizep[1]-3, sizep[0]])##
        ###############################################################
        
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
    #     print(model_dir)
        try:
            trans[i] = np.loadtxt(model_dir+'/trans')[:,1]# + x_o[:,2]*np.random.randn(1, x_o.shape[0])
        except:
            print('Trans: ', model_dir)
        if phaseR:
            try:
                phases = np.loadtxt(model_dir+'/phase')
                phase[i] = (phases[:,1:-2]/phases[:,-2:-1]).T #flatten(order='F')
            except:
                print('Phase: ', model_dir)
    
    if phaseR:
        return trans, phase
    else:
        return trans