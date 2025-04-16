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
import pickle
from modules import *

#### There should be at least two functions in this file:
####   - One called 'read_input' that returns a list with the names of the parameters, the prior bounds (as a list of tuples, only supports uniform priors for now), 
####     a list with all observation file names, the observation, the noise of the observation and the number of atmospheric layers.
####   - One called simulator that returns simulated spectra

def delta(o,m,s):  ### wtf is this???
    return sum((o-m)/s**2)/sum(1/s**2)

def scale(obs, model):
    scale = 1/len(obs)*(sum(obs-model))
    return scale

def read_input(args):
    
    os.system('cp '+args.input + ' '+args.output+'/input.dat')
    if args.input2!='aintnothinhere':
        os.system('cp '+args.input2 + ' '+args.output+'/input2.dat')

    args.input = args.output+'/input.dat'
    if args.input2!='aintnothinhere':
        args.input2 = args.output+'/input2.dat'
    
    inp = []
    with open(args.input, 'rb') as arcis_input:
        for lines in arcis_input:
            inp.append(str(lines).replace('\\n','').replace("b'","").replace("'", ""))
    clean_in = []
    for i in range(len(inp)):
        if inp[i]!='' and inp[i][0]!='*':
            clean_in.append(inp[i])
    
    which = []
        
    freeT=False

    parnames=[]
    prior_bounds=[]
    arcis_par=0
    log=[]
    i=0
    while i<len(clean_in):
        #I think this is just checking the number of atm layers to initialize PT arrays? Sounds inefficient and I should change it
        if 'nr=' in clean_in[i]:
            nr = int(clean_in[i][3:])
        elif 'nr =' in clean_in[i]:
            nr = int(clean_in[i][4:])
        if 'fitpar:keyword' in clean_in[i]:
            parnames.append(clean_in[i][16:-1])
            which.append(1)
            arcis_par+=1
            if clean_in[i+3]=='fitpar:log=.true.':
                prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                log.append(True)
            elif clean_in[i+3]=='fitpar:log=.false.':
                prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                log.append(False)
            i+=4
        # if 'fitpar:keyword'
        else:
            i+=1
    init2 = len(parnames)
            
    if args.input2 != 'aintnothinhere':
        print('Reading 2nd input file...')
        inp2 = []
        with open(args.input2, 'rb') as arcis_input2:
            for lines2 in arcis_input2:
                inp2.append(str(lines2).replace('\\n','').replace("b'","").replace("'", ""))
        clean_in2 = []
        for i in range(len(inp2)):
            if inp2[i]!='' and inp2[i][0]!='*':
                clean_in2.append(inp2[i])


        freeT=False
        
        parnames2=[]
        prior_bounds2=[]
        i=0
        while i<len(clean_in2):
            if 'fitpar:keyword' in clean_in2[i]:
                parnames2.append(clean_in2[i][16:-1])
                if clean_in2[i+3]=='fitpar:log=.true.':
                    prior_bounds2.append([np.log10(float(clean_in2[i+1][11:].replace('d','e'))), np.log10(float(clean_in2[i+2][11:].replace('d','e')))])
                elif clean_in2[i+3]=='fitpar:log=.false.':
                    prior_bounds2.append([float(clean_in2[i+1][11:].replace('d','e')), float(clean_in2[i+2][11:].replace('d','e'))])
                i+=4
                
            # if 'fitpar:keyword'
            else:
                i+=1
                
        assert args.n_global<=len(parnames2), 'Tsk tsk tsk... There\'s more global parameters than parameters *smh*'
        
        for i in range(args.n_global, len(parnames2)):
            parnames.append(parnames2[i])
            prior_bounds.append(prior_bounds2[i])
            which.append(2)
    
    if args.fit_frac:
        parnames.append('frac')
        log.append(False)
        prior_bounds.append([0.,1.])

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
    print('Loading observations...')
    logging.info('Loading observations...')
    nwvl = np.zeros(len(obs))
    for i in range(len(obs)):
        nwvl[i] = int(len(np.loadtxt(obs[i])[:,0]))
    l=[0]
    wvl_spec = np.zeros(int(sum(nwvl)))
    obs_spec = np.zeros(int(sum(nwvl)))
    noise_spec =np.zeros(int(sum(nwvl)))
    for j in range(len(obs)):
        phasej = np.loadtxt(obs[j])
        l.append(len(phasej))
        wvl_spec[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,0]
        obs_spec[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,1]
        noise_spec[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,2]
        
        
    ### Add scaling as parameter    
    if args.fit_scaling:
        for i in range(1, len(obs)):
            parnames.append(f'scaling_{i}')
            log.append(False)
            prior_bounds.append([1-args.max_scaling, 1+args.max_scaling])

    if args.fit_offset:
        for i in range(1, len(obs)):
            parnames.append(f'offset_{i}')
            log.append(False)
            prior_bounds.append([-args.max_offset, args.max_offset])

    ### Add vrot as parameter (km/s)
    if args.fit_vrot:
        parnames.append(f'vrot')
        log.append(False)
        prior_bounds.append([args.min_vrot, args.max_vrot])

    ### Add RV as parameter (km/s)
    if args.fit_RV:
        parnames.append(f'RV')
        log.append(False)
        prior_bounds.append([args.min_RV, args.max_RV])
    
    # if args.fit_offset:
    #     assert len(offsets)//2<len(obs), f'Are you sure you want more offsets ({len(offsets)}) than observations ({len(obs)})?'
    
    prior_bounds=np.array(prior_bounds)
        
    plot_info={}
    plot_info['parnames']=parnames
    plot_info['prior_bounds']=prior_bounds
    pickle.dump(plot_info, open(f'{args.output}/plot_info.p', 'wb'))
          
    return parnames, prior_bounds, obs, wvl_spec, obs_spec, noise_spec, nr, which, init2, nwvl, log, arcis_par

# def x2Ppoints(x, nTpoints):
#     Pmin=1e2
#     Pmax=1e-7
#     Ppoint = np.empty([len(x), nTpoints])
#     for j in range(len(x)):
#         Ppoint[j, 0] = 10**(np.log10(Pmin) + np.log10(Pmax/Pmin) * (1- x[j]**(1/nTpoints)))
#         for i in range(1, nTpoints):
#             Ppoint[j,i] = 10**(np.log10(Ppoint[j,i-1]) + np.log10(Pmax/Ppoint[j,i-1]) * (1- x[j]**(1/(nTpoints-i+1))))
#     return Ppoint

def simulator(fparameters, parnames, directory, r, input_file, input2_file, n_global, which, wvl_spec, nr, n, n_obs, size, nwvl, arcis_par, args):
    fname = directory+'/round_'+str(r)+str(n)+'_samples.dat'
    
    # if args.fit_offset:
    #     offsets=args.prior_offset.split(',')
    #     parameters=fparameters[:,:-len(offsets)//2]
    #     offset = fparameters[:,-len(offsets)//2:]
    # else:
        
    parameters=fparameters[:, :arcis_par]
    extra_pars=fparameters[:, arcis_par:]
    
    if input2_file!='aintnothinhere':
        print('Writing parametergridfile for 2nd limb')
        fname2 = directory+'/round_'+str(r)+str(n)+'_samples2.dat'
    
        parametergrid1 = parameters[:,:n_global]
        parametergrid2 = parameters[:,:n_global]            

        for i in range(n_global, len(which)):
            if which[i]==1:
                print('Appending to limb 1')
                parametergrid1=np.concatenate((parametergrid1, parameters[:,i:i+1]),axis=1)
            elif which[i]==2:
                print('Appending to limb 2')
                parametergrid2=np.concatenate((parametergrid2, parameters[:,i:i+1]),axis=1)
        
        # for i in range(len(parametergrid1)):
        #     if parametergrid1[i,n_global]<parametergrid2[i,n_global]:
        #         mT = parametergrid1[i,n_global]
        #         MT = parametergrid2[i,n_global]
        #         parametergrid1[i,n_global]=MT
        #         parametergrid2[i,n_global]=mT

        np.savetxt(fname, parametergrid1)
        np.savetxt(fname2, parametergrid2)
    else:
        parametergrid = parameters
        np.savetxt(fname, parametergrid)
        
    # print('Running ARCiS')
    # print('Computing '+str(len(parameters))+' models for process '+str(n))
    # logging.info('Running ARCiS')
    # logging.info('Computing '+str(len(parameters))+' models for process '+str(n))
    os.system('cd ; '+ARCiS + ' '+input_file + ' -o '+directory+'/round_'+str(r)+str(n)+'_out -s parametergridfile='+fname)
    if input2_file!='aintnothinhere':
        # print('Computing 2nd limb')
        # logging.info('Computing 2nd limb')
        os.system('cd ; '+ARCiS + ' '+input2_file + ' -o '+directory+'/round_'+str(r)+str(n)+'_limb2_out -s parametergridfile='+fname2)

    dirx = directory + '/round_'+str(r)+str(n)+'_out/'
    
    # print('All is good\n')
    arcis_spec1 = -1*np.ones([parameters.shape[0], size])
    # print('If you\'re seeing this, I\'m not replicating michiels issue\n')    
        
    print('Reading ARCiS output')
    # logging.info('Reading ARCiS output')
    
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
        except:
            print(f'Couldn\'t store P-T {model_dir}')
            logging.info(f'Couldn\'t store P-T {model_dir}')
        l=[0]
        for j in range(n_obs):
            # obsi = np.loadtxt(obs[i])
            if j+1<10:
                obsn = '/obs00'+str(j+1)
            elif j+1<100:
                obsn = '/obs0'+str(j+1)
            try:
                phasej = np.loadtxt(model_dir+obsn)[:,1]
            except:
                print(f'Couldn\'t store observation {model_dir}{obsn}')
                logging.info(f'Couldn\'t store observation {model_dir}{obsn}')
            l.append(len(phasej))
            
            if not(np.any(phasej<0) or np.any(np.isnan(phasej)) or np.any(np.isinf(phasej))):
                # print(phasej)
                arcis_spec1[i][sum(l[:j+1]):sum(l[:j+2])] = phasej #+off  
        
            
            
        tname = directory+'/T_round_'+str(r)+str(n)
        add = 'I'
        np.save(tname+'.npy', T)
        # if os.path.isfile(tname+'.npy') == False:
        #     np.save(tname+'.npy', T)
        # else:
        #     while os.path.isfile(tname+'.npy') == True:
        #         print(tname+'.npy already exists, saving as '+tname+add+'.npy')
        #         logging.info(tname+'.npy already exists, saving as '+tname+add+'.npy')
        #         tname = tname+add
        #         if os.path.isfile(tname+'.npy') == False:
        #             np.save(tname+'.npy', T)
        #             break
        np.save(directory+'/P.npy',P)
    
    if input2_file != 'aintnothinhere':
        dirx2 = directory + '/round_'+str(r)+str(n)+'_limb2_out/'

        arcis_spec2 = np.zeros([parameters.shape[0], size])

        # print('Reading ARCiS output')
        # logging.info('Reading ARCiS output')

        T2 = np.zeros([parameters.shape[0], nr])
        P2 = np.zeros(nr)

        for i in trange(parameters.shape[0]):
            if i+1<10:
                model_dir = dirx2 + 'model00000'+str(i+1)
            elif i+1<100:
                model_dir = dirx2 + 'model0000'+str(i+1)
            elif i+1<1000:
                model_dir = dirx2 + 'model000'+str(i+1)
            elif i+1<1e4:
                model_dir = dirx2 + 'model00'+str(i+1)
            try:
                PT2 = np.loadtxt(model_dir+'/mixingratios.dat')
                T2[i] = PT2[:,0]
                P2 = PT2[:,1]
                l=[0]
                for j in range(n_obs):
                    if j+1<10:
                        obsn = '/obs00'+str(j+1)
                    elif j+1<100:
                        obsn = '/obs0'+str(j+1)
                    phasej = np.loadtxt(model_dir+obsn)[:,1]
                    l.append(len(phasej))
                    if np.any(phasej<0) or np.any(np.isnan(phasej)) or np.any(np.isinf(phasej)):
                        arcis_spec2[i][sum(l[:j+1]):sum(l[:j+2])] = -1*np.ones_like(phasej)
                    else:
                        arcis_spec2[i][sum(l[:j+1]):sum(l[:j+2])] = phasej
            except:
                print('Couldn\'t store model ', model_dir)
                logging.info('Couldn\'t store model ', model_dir)

            tname2 = directory+'/T_limb2_round_'+str(r)+str(n)
            add = 'I'
            np.save(tname2+'.npy', T2)
            np.save(directory+'/P.npy',P2)
        
        if args.fit_frac:
            fracs = parameters[:,-1]
        else:
            fracs=0.5*np.ones(parameters.shape[0])
        arcis_spec=np.zeros([parameters.shape[0], size])
        for i in range(parameters.shape[0]):
            if sum(arcis_spec1[i])!=0 and sum(arcis_spec2[i])!=0:
                if args.binary:
                    arcis_spec[i]=arcis_spec1[i] + arcis_spec2[i]
                else:
                    arcis_spec[i]=fracs[i]*arcis_spec1[i] + (1-fracs[i])*arcis_spec2[i]
    else:
        arcis_spec=arcis_spec1
        

    if args.fit_scaling and n_obs>1:
        add_log('Adding scalings...')
        for i in range(parameters.shape[0]):
            scaling=[]
            for o in range(len(parnames[arcis_par:])):
                if 'scaling' in parnames[arcis_par:][o]:
                    scaling.append(extra_pars[i][o])
            for j in range(1, n_obs):
                arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] *= scaling[j-1]
    
    
    if args.fit_offset and n_obs>1:
        add_log('Adding offsets...')
        for i in range(parameters.shape[0]):
            offset=[]
            for o in range(len(parnames[arcis_par:])):
                if 'offset' in parnames[arcis_par:][o]:
                    offset.append(extra_pars[i][o])
            for j in range(1, n_obs):
                arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] += offset[j-1]
    
    if args.fit_vrot:
        add_log('Adding rotational broadening...')
        for o in range(len(parnames[arcis_par:])):
            if 'vrot' in parnames[arcis_par:][o]:
                vrots = extra_pars[:,o]
        for i in range(parameters.shape[0]):
            for j in range(1, n_obs):
                unbroadened = arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
                wvl = wvl_spec[int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
                broadened = rot_int_cmj(wvl, unbroadened, vrots[i])
                arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] = broadened
    elif args.vrot>0:
        for j in range(1, n_obs):
            unbroadened = arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
            wvl = wvl_spec[int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
            broadened = rot_int_cmj(wvl, unbroadened, args.vrot)
            arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] = broadened

    if args.fit_RV:
        add_log('Adding rotational broadening...')
        for o in range(len(parnames[arcis_par:])):
            if 'RV' in parnames[arcis_par:][o]:
                vrots = extra_pars[:,o]
        for i in range(parameters.shape[0]):
            for j in range(1, n_obs):
                unbroadened = arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
                wvl = wvl_spec[int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
                broadened = rot_int_cmj(wvl, unbroadened, vrots[i])
                arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] = broadened
    elif args.RV>0:
        for j in range(1, n_obs):
            unshifted = arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
            wvl = wvl_spec[int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))]
            shifted = dopplerShift(wvl, unbroadened, args.vrot)
            arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] = broadened
    # if args.test_box:
        
    return arcis_spec