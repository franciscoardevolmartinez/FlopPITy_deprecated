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
from multiprocessing import Process, Pool

#### There should be at least two functions in this file:
####   - One called 'read_input' that returns a list with the names of the parameters, the prior bounds (as a list of tuples, only support uniform priors for now), 
####     a list with all observation file names, the observation, the noise of the observation and the number of atmospheric layers.
####   - One called simulator that returns simulated spectra


def read_input(args):
    
    os.system('cp '+args.input + ' '+args.output+'/input_ARCiS.dat')

    args.input = args.output+'/input_ARCiS.dat'
    
    inp = []
    with open(args.input, 'rb') as arcis_input:
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
            elif args.twoterms:
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

def simulator(parameters, directory, r, input_file, nr, n, n_obs, size):
    fname = directory+'/round_'+str(r)+str(n)+'_samples.dat'
    
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

'''
def compute(params, nprocesses, output, arginput, ynorm, r, nr, obs, obs_spec):
    samples_per_process = len(params)//nprocesses

    print('Samples per process: ', samples_per_process)
    freeT=False
    parargs=[]
    
    # Delete this and just input the un-transformed parameters.
    
    # if ynorm:
    #     params=yscaler.inverse_transform(np_theta)
    # else:
    #     params = np_theta

    # if ynorm:
    #     params=yscaler.inverse_transform(np_theta)
    # else:
    #     params = np_theta  #### QUICK FIX, MAKE IT GOOD!!!
    # if freeT:
    #     for i in range(nprocesses-1):
    #         parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints, nr,i, len(obs), len(obs_spec)))
    #     parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints, nr,args.processes-1, len(obs), len(obs_spec)))
    # else:
    for i in range(nprocesses-1):
        parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], output, r, arginput, nr, i, len(obs), len(obs_spec)))
    parargs.append((params[(nprocesses-1)*samples_per_process:], output, r, arginput, nr, nprocesses-1, len(obs), len(obs_spec)))

    # tic=time()
    with Pool(processes = nprocesses) as pool:
        # Simulator is function that returns spectra (needs to be same units as observation)
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    for j in range(nprocesses):
        os.system('mv '+output + '/round_'+str(r)+str(j)+'_out/log.dat '+output +'/ARCiS_logs/log_'+str(r)+str(j)+'.dat')
        os.system('rm -rf '+output + '/round_'+str(r)+str(j)+'_out/')
        os.system('rm -rf '+output + '/round_'+str(r)+str(j)+'_samples.dat')
# print('Tim/fo(('Time elapsed: ', time()-tic))
    
    return arcis_spec

def compute_2term(np_theta):
    samples_per_process = 2*len(np_theta)//args.processes

    print('Samples per process: ', samples_per_process)

    parargs=[]
    if args.ynorm:
        params1=yscaler.inverse_transform(np_theta)
    else:
        params1 = np_theta
    print(params1.shape)
    params= np.zeros([2*params1.shape[0], (params1.shape[1]-2)//2+2])
    params[::2,:-2] = params1[:,:-2][:,::2]
    params[1::2,:-2] = params1[:,:-2][:,1::2]
    params[::2,-2:] = params1[:,-2:]
    params[1::2,-2:] = params1[:,-2:]
    # params=params1.reshape([len(np_theta[0])//2, 2*samples_per_round])
    if freeT:
        for i in range(args.processes-1):
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints,nr, i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints,nr, args.processes-1, len(obs), len(obs_spec)))
    else:
        for i in range(args.processes-1):
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, 0,nr, i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, 0, nr,args.processes-1, len(obs), len(obs_spec)))

    # tic=time()
    with Pool(processes = args.processes) as pool:
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    # print('Time elapsed: ', time()-tic)
    # logging.info(('Time elapsed: ', time()-tic))
    
    arcis_spec_2 = np.zeros([len(np_theta), len(obs_spec)])
    for i in range(len(np_theta)):
        arcis_spec_2[i]=np.mean(arcis_spec[2*i:2*i+2], axis=0)
    
    return arcis_spec_2
'''