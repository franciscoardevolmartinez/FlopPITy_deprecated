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
    with open(fname, "wb") as file_parameters:
        np.savetxt(file_parameters, parameters)

    print('Running ARCiS')
    print('Computing '+str(len(parameters))+' models for process '+str(n))
    logging.info('Running ARCiS')
    os.system('cd .. ; '+ARCiS + ' '+input_file + ' -o '+directory+'round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)

    dirx = directory + 'round_'+str(r)+str(n)+'_out/'
    
    sizet = np.loadtxt(dirx+'model000001/obs001').shape[0]
    emis = np.zeros([parameters.shape[0], sizet])
    
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
            emis[i] = np.loadtxt(model_dir+'/obs001')[:,1]
        except:
            print('Emis: ', model_dir)
        
        return emis
