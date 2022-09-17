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

def simulator(parameters, directory, r, input_file, obs_file, n):

    fname = directory+'round_'+str(r)+str(n)+'_samples.dat'
    np.savetxt(fname, parameters)

    print('Running ARCiS')
    logging.info('Running ARCiS')
    os.system('cd .. ; '+ARCiS + ' '+input_file + ' -o '+directory+'round_'+str(r)+str(n)+'_out -s parametergridfile='+fname+' -s obs01:file='+obs_file)
    
    x_o = np.loadtxt(obs_file)
    
    X = np.zeros([parameters.shape[0], x_o.shape[0]])

    dirx = directory + 'round_'+str(r)+str(n)+'_out/'
    
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
            X[i] = np.loadtxt(model_dir+'/trans')[:,1]# + x_o[:,2]*np.random.randn(1, x_o.shape[0])
        except:
            print('Trans: ', model_dir)
    
    return X