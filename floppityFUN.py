#!/usr/bin/env python3

# import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from time import time
import logging
from typing import List, Optional, Tuple, Union
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding
from simulator import *

### (Log) likelihood. Necessary to compute (log) evidence
def likelihood(obs, err, x):
    L = 0
    for i in range(len(obs)):
        L += -np.log(np.sqrt(2*np.pi)*err[i]) + (-(obs[i]-x[i])**2/(2*err[i]**2))
    return L

def evidence(posterior, prior, samples, Y, obs, err, do_pca, xnorm, rem_mean, xscaler, pca):
    L = np.empty(len(samples))
    for j in range(len(samples)):
        L[j] = likelihood(obs, err, samples[j])
    default_x = xscaler.transform(pca.transform(rem_mean.transform(obs.reshape(1,-1))))
    P = posterior.log_prob(torch.tensor(Y), x=default_x)
    pi = prior.log_prob(torch.tensor(Y))
    logZ =np.empty(3)
    logZ[0] = np.median(-(P-pi-L).detach().numpy())
    logZ[2] = np.percentile(-(P-pi-L).detach().numpy(), 84)
    logZ[1] = np.percentile(-(P-pi-L).detach().numpy(), 16)
    return logZ

def evidence_w_all(posterior, prior, samples, Y, obs, err, do_pca, xnorm):
    L = np.empty(len(samples[0])*len(samples))
    for j in range(len(samples)):
        for k in range(len(samples[j])):
            L[j*len(samples[j])+k] = likelihood(obs, err, samples[j][k])
    if do_pca:
        if xnorm:
            default_x = xscaler.transform(pca.transform(obs.reshape(1,-1)))
        else:
            default_x = pca.transform(obs.reshape(1,-1))
    else:
        if xnorm:
            default_x = xscaler.transform(obs.reshape(1,-1))
        else:
            default_x = obs.reshape(1,-1)
    P = np.empty(len(samples[0])*len(samples))
    pi = np.empty(len(samples[0])*len(samples))
    for l in range(len(samples)):
        P[len(samples[0])*l:len(samples[0])*(l+1)] = posterior.log_prob(torch.tensor(Y[l]), x=default_x).detach().numpy()
        pi[len(samples[0])*l:len(samples[0])*(l+1)] = prior.log_prob(torch.tensor(Y[l])).detach().numpy()
    logZ =np.empty(3)
    logZ[0] = np.median(-(P-pi-L))
    logZ[2] = np.percentile(-(P-pi-L), 84)
    logZ[1] = np.percentile(-(P-pi-L), 16)
    return logZ

def unroll_embed_hypers(embed_hypers, embed_size):
    output_dims=embed_size.split(',')
    for i in range(len(output_dims)):
        output_dims[i]=int(output_dims[i])
        
    hypers = embed_hypers.split(',')
    
    num_conv_layers = int(hypers[0])
    out_channels_per_layer = []
    for i in range(num_conv_layers):
        out_channels_per_layer.append(int(hypers[1])*(i+1))
    num_linear_layers = int(hypers[2])
    num_linear_units = int(hypers[3])
    kernel_size = int(hypers[4])
        
    return num_conv_layers, out_channels_per_layer, num_linear_layers, num_linear_units, kernel_size, output_dims
    
class multiNet(nn.Module):
    def __init__(
        self,
        nwvl,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: List = [6, 12],
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
        num_linear_units: int = 50,
        output_dim: List = [20, 20],
        kernel_size: int = 5,
        pool_kernel_size: int = 2,
    ):
        super(multiNet, self).__init__()
        
        self.nets = []
        self.nwvl = nwvl
        for i in range(len(self.nwvl)):
            self.nets.append(CNNEmbedding(input_shape=(int(nwvl[i]),),
                                         out_channels_per_layer=out_channels_per_layer,
                                         num_conv_layers=num_conv_layers,
                                         num_linear_layers=num_linear_layers,
                                         num_linear_units=num_linear_units,
                                         output_dim=output_dim[i],
                                         kernel_size=kernel_size,
                                         pool_kernel_size=pool_kernel_size))
            
    def forward(self, X: Tensor) -> Tensor:
        x=[]
        for i in range(len(self.nwvl)):
            x.append(X[:, int(sum(self.nwvl[:i])):int(sum(self.nwvl[:i+1]))])
            x[i] = self.nets[i](x[i])
                        
        return torch.cat(x, dim=1)
    
### Display 1D marginals in console
def post2txt(post, parnames, prior_bounds, nbins=20, a=33, b=67):
    D = post.shape[1]
    ls = [len(k) for k in parnames]
    for j in range(D):
        edges = [prior_bounds[j][0]]
        binwidth=(prior_bounds[j][1]-prior_bounds[j][0])/nbins
        for i in range(nbins):
            edges.append(edges[-1]+binwidth)
        try:
            counts, _ = np.histogram(post[:,j], bins=edges)
        except:
            counts, _ = np.histogram(post[:,j], bins=edges[::-1])
        a_=np.percentile(counts[counts>0],a)
        b_=np.percentile(counts[counts>0],b)
        oneD=['|']+nbins*[' ']+['|']
        for i in range(nbins):
            if counts[i]>b_:
                oneD[1+i]='*'
            elif counts[i]>a_:
                oneD[1+i]='.'
        print(parnames[j]+(max(ls)-ls[j])*' '+' -> '  +''.join(oneD)+
              '  /  '+str(round(np.median(post[:,j]),2))+'  ('+
              str(round(np.percentile(post[:,j],16),2))+', '+
              str(round(np.percentile(post[:,j],84),2))+')')
        logging.info(parnames[j]+(max(ls)-ls[j])*' '+' -> '  +''.join(oneD)+
              '  /  '+str(round(np.median(post[:,j]),2))+'  ('+
              str(round(np.percentile(post[:,j],16),2))+', '+
              str(round(np.percentile(post[:,j],84),2))+')')
    
### Parameter transformer
class Normalizer():
    def __init__(self, prior_bounds):
        self.bounds = prior_bounds
        
    def transform(self, Y):
        assert len(self.bounds) == Y.shape[1], 'Dimensionality of prior and parameters doesn\'t match!'
        Yt = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            Yt[:,i] = 2*(Y[:,i] - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0])-1
        return Yt
    
    def inverse_transform(self, Y):
        assert len(self.bounds) == Y.shape[1], 'Dimensionality of prior and parameters doesn\'t match!'
        Yi = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            Yi[:,i] = (Y[:,i]+1)*(self.bounds[i][1] - self.bounds[i][0])/2 + self.bounds[i][0]
        return Yi

class sigma_res_scale():
    def __init__(self):
        return
    
    def transform(data,obs,noise):
        return (data - obs)/noise
        
    def inverse_transform(sigma_res,obs,noise):
        return sigma_res*noise+obs
              
class res_scale():
    def __init__(self):
        return
    
    def transform(data, obs):
        return data - obs
    
    def inverse_transform(res, obs):
        return res + obs

class do_nothing():
    def __init__(self):
        return
        
    def transform(self, data):
        return data
    
    def fit(self, data):
        # This does nothing but replicates other preprocessing classes
        return 
    
    def inverse_transform(self, data):
        return data
    
class rm_mean():
    def __init__(self):
        return
        
    def transform(self, arcis_spec):
        normed = np.empty([arcis_spec.shape[0],arcis_spec.shape[1]+1])
        logging.info('Removing the mean')
        print('Removing the mean')
        for i in trange(len(arcis_spec)):
            xbar=np.mean(arcis_spec[i])
            normed[i][:-1] = arcis_spec[i]-xbar
            normed[i][-1] = xbar
    
        return normed
    
    def inverse_transform(self, normed):
        arcis_spec = np.empty([normed.shape[0],normed.shape[1]-1])
        logging.info('Adding back the mean')
        print('Adding back the mean')
        for i in trange(len(normed)):
            xbar=normed[i][-1]
            arcis_spec[i] = normed[i]+xbar
    
        return arcis_spec

### COMPUTE FORWARD MODELS FROM NORMALISED PARAMETERS
def compute(params, nprocesses, output, arginput, arginput2, n_global, which, ynorm, r, nr, obs, obs_spec,nwvl,args):
    
    
    samples_per_process = len(params)//nprocesses*np.ones(nprocesses, dtype=int)
    rem = int(len(params)-sum(samples_per_process))
    samples_per_process[:rem]+=np.ones(rem,dtype=int)    

    print(f'Samples per process: {samples_per_process[0]}')
    freeT=False
    parargs=[]
    # if ynorm:
    #     params=yscaler.inverse_transform(np_theta)
    # else:
    #     params = np_theta
    # if freeT:
    #     for i in range(nprocesses-1):
    #         parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints, nr,i, len(obs), len(obs_spec)))
    #     parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints, nr,args.processes-1, len(obs), len(obs_spec)))
    # else:
    for i in range(nprocesses):
        parargs.append((params[i*samples_per_process[i]:(i+1)*samples_per_process[i]], output, r, arginput, arginput2, n_global, which, nr, i, len(obs), len(obs_spec),nwvl, args))
    # parargs.append((params[(nprocesses-1)*samples_per_process:], output, r, arginput, arginput2, n_global, which, nr, nprocesses-1, len(obs), len(obs_spec),nwvl, args))

    # tic=time()
    print('Running ARCiS')
    # print(f'Computing {str(samples_per_process)} models per process')
    logging.info('Running ARCiS')
    logging.info('Computing '+str(samples_per_process[0])+' models per process ')
    with Pool(processes = nprocesses) as pool:
        # Simulator is function that returns spectra (needs to be same units as observation)
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    for j in range(nprocesses):
        os.system('mv '+output + '/round_'+str(r)+str(j)+'_out/log.dat '+output +'/ARCiS_logs/log_'+str(r)+str(j)+'.dat')
        os.system('rm -rf '+output + '/round_'+str(r)+str(j)+'_out/')
        os.system('rm -rf '+output + '/round_'+str(r)+str(j)+'_limb2_out/')
        os.system('rm -rf '+output + '/round_'+str(r)+str(j)+'_samples.dat')
# print('Tim/fo(('Time elapsed: ', time()-tic))
    
    return arcis_spec

def compute_2term(params1,nprocesses, output, arginput, ynorm, r, nr, obs, obs_spec):
    samples_per_process = 2*len(params1)//nprocesses

    print('Samples per process: ', samples_per_process)

    parargs=[]
    # if args.ynorm:
    #     params1=yscaler.inverse_transform(np_theta)
    # else:
    #     params1 = np_theta
    print(params1.shape)
    params= np.zeros([2*params1.shape[0], (params1.shape[1]-5)//2+5])
    print(params.shape)
    params[::2,3:-2] = params1[:,3:-2][:,::2]
    params[1::2,3:-2] = params1[:,3:-2][:,1::2]
    params[::2,0:3] = params1[:,0:3]
    params[1::2,0:3]= params1[:,0:3]
    params[::2,-2:] = params1[:,-2:]
    params[1::2,-2:] = params1[:,-2:]
    print
    # params=params1.reshape([len(np_theta[0])//2, 2*samples_per_round])
    # if freeT:
    #     for i in range(args.processes-1):
    #         parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints,nr, i, len(obs), len(obs_spec)))
    #     parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints,nr, args.processes-1, len(obs), len(obs_spec)))
    # else:
    for i in range(nprocesses-1):
        parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], output, r, arginput, nr, i, len(obs), len(obs_spec)))
    parargs.append((params[(nprocesses-1)*samples_per_process:], output, r, arginput,  nr,nprocesses-1, len(obs), len(obs_spec)))

    # tic=time()
    with Pool(processes = nprocesses) as pool:
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    # print('Time elapsed: ', time()-tic)
    # logging.info(('Time elapsed: ', time()-tic))
    
    arcis_spec_2 = np.zeros([len(params1), len(obs_spec)])
    for i in range(len(params1)):
        arcis_spec_2[i]=np.mean(arcis_spec[2*i:2*i+2], axis=0)
    
    return arcis_spec_2
    
### Preprocessing for spectra and parameters
def preprocess(np_theta, arcis_spec, r, samples_per_round, obs_spec,noise_spec,naug,do_pca, n_pca, xnorm, rem_mean, output, device, args):
    theta_aug = torch.tensor(np.repeat(np_theta, naug, axis=0), dtype=torch.float32, device=device)
    arcis_spec_aug = np.repeat(arcis_spec,naug,axis=0) + noise_spec*np.random.randn(samples_per_round*naug, obs_spec.shape[0]) #surely this can be changed to arcis_spec.shape ?? Apparently not

    if do_pca:
        pca = PCA(n_components=n_pca)
    else:
        pca = do_nothing()
    
    if xnorm:
        xscaler = StandardScaler()
    else:
        xscaler = do_nothing()
        
    if rem_mean:
        rem_mean=rm_mean()
    else:
        rem_mean=do_nothing()
    
    if r==0:
        
        pca.fit(rem_mean.transform(arcis_spec))
        with open(output+'/pca.p', 'wb') as file_pca:
            pickle.dump(pca, file_pca)
        xscaler.fit(pca.transform(rem_mean.transform(arcis_spec)))
        with open(output+'/xscaler.p', 'wb') as file_xscaler:
            pickle.dump(xscaler, file_xscaler)
            
    pca=pickle.load(open(output+'/pca.p', 'rb'))
    xscaler=pickle.load(open(output+'/xscaler.p', 'rb'))
    if args.res_scale:
        print('Training on residuals')
        x_f = torch.tensor(xscaler.transform(pca.transform(sigma_res_scale.transform(arcis_spec_aug, obs_spec.reshape(1,-1), noise_spec.reshape(1,-1)))), dtype=torch.float32, device=device)
    else:
        x_f = torch.tensor(xscaler.transform(pca.transform(rem_mean.transform(arcis_spec_aug))), dtype=torch.float32, device=device)

    return theta_aug, x_f, xscaler, pca