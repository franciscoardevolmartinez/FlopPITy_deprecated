#
#!/usr/bin/env python3

ARCiS = 'ARCiS'
# ARCiS = 'ARCiS_phot'
# ARCiS = 'ARCiS_clouds'

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
from tqdm import trange
from multiprocessing import Process

def read_input(inputfile, twoterms):
    inp = []
    with open(inputfile, 'rb') as arcis_input:
        for lines in arcis_input:
            inp.append(str(lines).replace('\\n','').replace("b'","").replace("'", ""))
    clean_in = []
    for i in range(len(inp)):
        if inp[i]!='' and inp[i][0]!='*':
            clean_in.append(inp[i])

    freeT=False

    parnames=[]
    prior_bounds=[]
    i=0
    while i<len(clean_in):
        #I think this is just checking the number of atm layers to initialize PT arrays? Sounds inefficient and I should change it
        if 'nr=' in clean_in[i]:
            nr = int(clean_in[i][3:])
        elif 'nr =' in clean_in[i]:
            nr = int(clean_in[i][4:])
        if 'fitpar:keyword' in clean_in[i]:
            if clean_in[i][16:-1] == 'tprofile':
                freeT=True
                nTpoints = int(clean_in[i+1][9:])
                for j in range(nTpoints):
                    parnames.append('dTpoint00'+str(1+j))
                    prior_bounds.append([-4/7, 4/7])
    #            for k in range(nTpoints):
    #                parnames.append('Ppoint00'+str(1+k))
                parnames.append('x')
                prior_bounds.append([0,1])
                i+=3
            elif twoterms:
                if clean_in[i][16:-1]=='Rp' or clean_in[i][16:-1]=='loggP':
                    parnames.append(clean_in[i][16:-1])
                    if clean_in[i+3]=='fitpar:log=.true.':
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                    elif clean_in[i+3]=='fitpar:log=.false.':
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                else:
                    parnames.append(clean_in[i][16:-1]+'_1')
                    parnames.append(clean_in[i][16:-1]+'_2')
                    if clean_in[i+3]=='fitpar:log=.true.':
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                    elif clean_in[i+3]=='fitpar:log=.false.':
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                i+=4
            else:
                parnames.append(clean_in[i][16:-1])
                if clean_in[i+3]=='fitpar:log=.true.':
                    prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                elif clean_in[i+3]=='fitpar:log=.false.':
                    prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                i+=4
        else:
            i+=1

    prior_bounds=np.array(prior_bounds)

    ### READ OBSERVATIONS
    obs=[]
    i=0
    obsn=1
    while i<len(clean_in):
        if 'obs'+str(obsn)+':file' in clean_in[i]:
            obs.append(clean_in[i][len('obs'+str(obsn)+':file')+2:-1])
            obsn+=1
        i+=1

    ### 1. Load observation/s
    print('Loading observations... ')
    logging.info('Loading observations...')
    nwvl = np.zeros(len(obs))
    for i in range(len(obs)):
        nwvl[i] = len(np.loadtxt(obs[i])[:,0])
    l=[0]
    obs_spec = np.zeros(int(sum(nwvl)))
    noise_spec =np.zeros(int(sum(nwvl)))
    for j in range(len(obs)):
        phasej = np.loadtxt(obs[j])
        l.append(len(phasej))
        obs_spec[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,1]
        noise_spec[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,2]
          
    return parnames, prior_bounds,obs, obs_spec,noise_spec, nr

def x2Ppoints(x, nTpoints):
    Pmin=1e2
    Pmax=1e-7
    Ppoint = np.empty([len(x), nTpoints])
    for j in range(len(x)):
        Ppoint[j, 0] = 10**(np.log10(Pmin) + np.log10(Pmax/Pmin) * (1- x[j]**(1/nTpoints)))
        for i in range(1, nTpoints):
            Ppoint[j,i] = 10**(np.log10(Ppoint[j,i-1]) + np.log10(Pmax/Ppoint[j,i-1]) * (1- x[j]**(1/(nTpoints-i+1))))
    return Ppoint

def simulator(parameters, directory, r, input_file, freeT, nTpoints, nr, n, n_obs, size):
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
    os.system('cd ; '+ARCiS + ' '+input_file + ' -o '+directory+'/round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)

    dirx = directory + '/round_'+str(r)+str(n)+'_out/'
        
    arcis_spec = np.zeros([parameters.shape[0], size])
        
    print('Reading ARCiS output')
    logging.info('Reading ARCiS output')
    
    T = np.zeros([parameters.shape[0], nr])
    P = np.zeros(nr)
    
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
            logging.info('Couldn\'t store model ', model_dir)
        
        np.save(directory+'/T_round_'+str(r)+str(n)+'.npy',T)
        np.save(directory+'/P.npy',P)
    
    return arcis_spec
