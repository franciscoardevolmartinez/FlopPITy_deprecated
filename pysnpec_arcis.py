#!/usr/bin/env python3

#ARCiS = 'ARCiS'

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.inference import SNPE_C
from time import time
import logging
import os
from tqdm import trange
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from corner import corner
from multiprocessing import Process, Pool
from simulator import simulator

supertic = time()

### PARSE COMMAND LINE ARGUMENTS ###

def parse_args():
    parser = argparse.ArgumentParser(description=('Train SNPE_C'))
    parser.add_argument('-input', type=str, help='ARCiS input file for the retrieval')
    parser.add_argument('-output', type=str, default='output/', help='Directory to save output')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: CPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    parser.add_argument('-samples_per_round', type=int, default=1000, help='Number of samples to draw for training each round. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=32)
    parser.add_argument('-transforms', type=int, default=5)
    parser.add_argument('-bins', type=int, default=10)
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-ynorm', action='store_false')
    parser.add_argument('-xnorm', action='store_false')
    parser.add_argument('-Ztol', type=float, default=0.05)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-nrepeat', type=int, default=3)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-atoms', type=int, default=10)
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-dont_reject', action='store_false')
    return parser.parse_args()

### (Log) likelihood. Necessary to compute (log) evidence
def likelihood(obs, err, x):
    L = 0
    for i in range(len(obs)):
        L += -np.log(np.sqrt(2*np.pi)*err[i]) + (-(obs[i]-x[i])**2/(2*err[i]**2))
    return L

def evidence(posterior, prior, samples, Y, obs, err):
    L = np.empty(len(samples))
    for j in range(len(samples)):
        L[j] = likelihood(obs, err, samples[j])
    default_x = xscaler.transform(obs.reshape(1,-1))
    P = posterior.log_prob(torch.tensor(Y), x=default_x)
    pi = prior.log_prob(torch.tensor(Y))
    logZ =np.empty(3)
    logZ[0] = np.median(-(P-pi-L).detach().numpy())
    logZ[2] = np.percentile(-(P-pi-L).detach().numpy(), 84)
    logZ[1] = np.percentile(-(P-pi-L).detach().numpy(), 16)
    return logZ
    
### Parameter transformer
class Normalizer():
    def __init__(self, prior_bounds):
        self.bounds = prior_bounds
        
    def transform(self, Y):
        assert len(prior_bounds) == Y.shape[1], 'Dimensionality of prior and parameters doesn\'t match!'
        Yt = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            Yt[:,i] = 2*(Y[:,i] - self.bounds[i][0])/(self.bounds[i][1] - self.bounds[i][0])-1
        return Yt
    
    def inverse_transform(self, Y):
        assert len(prior_bounds) == Y.shape[1], 'Dimensionality of prior and parameters doesn\'t match!'
        Yi = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            Yi[:,i] = (Y[:,i]+1)*(self.bounds[i][1] - self.bounds[i][0])/2 + self.bounds[i][0]
        return Yi

work_dir = os.getcwd()+'/'

args = parse_args()

### COMPUTE FORWARD MODELS FROM NORMALISED PARAMETERS
def compute(np_theta):
    samples_per_process = len(np_theta)//args.processes

    print('Samples per process: ', samples_per_process)

    parargs=[]
    if args.ynorm:
        params=yscaler.inverse_transform(np_theta)
    else:
        params = np_theta
    if freeT:
        for i in range(args.processes-1):
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints, i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints, args.processes-1, len(obs), len(obs_spec)))
    else:
        for i in range(args.processes-1):
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, 0, i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, 0, args.processes-1, len(obs), len(obs_spec)))

    tic=time()
    with Pool(processes = args.processes) as pool:
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    print('Time elapsed: ', time()-tic)
    logging.info(('Time elapsed: ', time()-tic))
    return arcis_spec

p = Path(args.output)
p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=args.output+'/log.log', filemode='a', format='%(asctime)s %(message)s', 
                datefmt='%H:%M:%S', level=logging.DEBUG)

logging.info('Initializing...')

logging.info('Command line arguments: '+ str(args))
print('Command line arguments: '+ str(args))

device = args.device

os.system('cp '+args.input + ' '+args.output+'input_arcis.dat')

args.input = args.output+'input_arcis.dat'

### READ PARAMETERS
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

### Preprocessing for spectra and parameters
def preprocess(np_theta, arcis_spec):
    theta_aug = torch.tensor(np_theta, dtype=torch.float32, device=device)
    arcis_spec_aug = arcis_spec + noise_spec*np.random.randn(samples_per_round, obs_spec.shape[0])
    global xscaler
    if r==0:
        if args.xnorm:
            xscaler = StandardScaler().fit(arcis_spec)
            with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                pickle.dump(xscaler, file_xscaler)

    if args.xnorm:
        x_f = xscaler.transform(arcis_spec_aug)
    else:
        x_f = arcis_spec_aug

    x = torch.tensor(x_f, dtype=torch.float32, device=device)

    return theta_aug, x

### TRANSFORM PRIOR
if args.ynorm:
    yscaler = Normalizer(prior_bounds)
    with open(args.output+'/yscaler.p', 'wb') as file_yscaler:
        pickle.dump(yscaler, file_yscaler)

    prior_min = torch.tensor(yscaler.transform(prior_bounds[:,0].reshape(1, -1)).reshape(-1))
    prior_max = torch.tensor(yscaler.transform(prior_bounds[:,1].reshape(1, -1)).reshape(-1))
else:
    prior_min = torch.tensor(prior_bounds[:,0].reshape(1,-1))
    prior_max = torch.tensor(prior_bounds[:,1].reshape(1,-1))

prior = utils.BoxUniform(low=prior_min.to(device, non_blocking=True), high=prior_max.to(device, non_blocking=True), device=device)

num_rounds = args.num_rounds

neural_posterior = posterior_nn(model='nsf', hidden_features=args.hidden, num_transforms=args.transforms, num_bins=args.bins, num_blocks=args.blocks,
                                z_score_x='none', z_score_y='none', use_batch_norm=True, dropout_probability=args.dropout)

inference = SNPE_C(prior = prior, density_estimator=neural_posterior, device=device)

samples_per_round = args.samples_per_round


print('Training multi-round inference')
logging.info('Training multi-round inference')

proposal=prior
posteriors=[]

samples=[]

logZs = []

np_theta = {}
arcis_spec = {}

r=0
repeat=0

if args.resume:
    print('Reading files from previous run...')
    logging.info('Reading files from previous run...')
    np_theta = pickle.load(open(args.output+'/Y.p', 'rb'))
    arcis_spec = pickle.load(open(args.output+'/arcis_spec.p', 'rb'))
    r=len(arcis_spec.keys())
    posteriors=torch.load(args.output+'/posteriors.pt')
    proposal=posteriors[-1]
    samples=pickle.load(open(args.output+'/samples.p', 'rb'))
    logZs=pickle.load(open(args.output+'/evidence.p', 'rb'))
    inference=pickle.load(open(args.output+'/inference.p', 'rb'))
    xscaler=pickle.load(open(args.output+'/xscaler.p', 'rb'))
    yscaler=pickle.load(open(args.output+'/yscaler.p', 'rb'))
    num_rounds+=r

while r<num_rounds:
    print('\n')
    print('\n **** Training round ', r)
    logging.info('#####  Round '+str(r)+'  #####')
    
    ##### DRAW SAMPLES
    logging.info('Drawing '+str(samples_per_round)+' samples')
    print('Samples per round: ', samples_per_round)
    theta = proposal.sample((samples_per_round,))
    np_theta[r] = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])

    if args.ynorm:
        post_plot = yscaler.inverse_transform(np_theta[r])
    else:
        post_plot = np_theta[r]

    fig1 = corner(post_plot, color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
    with open(args.output+'corner_'+str(r)+'.jpg', 'wb') as file_corner:
        plt.savefig(file_corner, bbox_inches='tight')
    plt.close('all')

    #### COMPUTE MODELS
    arcis_spec[r] = compute(np_theta[r])

    for j in range(args.processes):
        os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
            
    ### COMPUTE EVIDENCE
    if r>0:
        logging.info('Computing evidence...')
        logZ = evidence(posteriors[-1], prior, arcis_spec[r], np_theta[r], obs_spec, noise_spec)
        print('\n')
        print('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        logging.info('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        print('\n')
        logZs.append(logZ)
    
    if args.dont_reject and r>1 and logZs[-1][0]<logZs[-2][0]:
        # If evidence doesn't improve we repeat last step
        repeat+=1
        print('Round rejected, repeating previous round. This round has been rejected '+str(repeat)+' times.')
        logging.info('Round rejected, repeating previous round. This round has been rejected '+str(repeat)+' times.')
        logZs.pop(-1)
        posteriors.pop(-1)
        del arcis_spec[r]
        del np_theta[r]
        # theta_aug, x = preprocess(np_theta[r-1], arcis_spec[r-1])
        reject=True
        if repeat>args.nrepeat:
            print('This round has been rejected the maximum number of times. Ending inference.')
            logging.info('This round has been rejected the maximum number of times. Ending inference.')
            break
    else:
        repeat=0
        if r>0:
            with open(args.output+'/evidence.p', 'wb') as file_evidence:
                print('Saving evidence...')
                logging.info('Saving evidence...')
                pickle.dump(logZs, file_evidence)
        ### PREPROCESS DATA
        theta_aug, x = preprocess(np_theta[r], arcis_spec[r])
        reject=False
        if r>1 and abs(logZs[-1][0]-logZs[-2][0])<args.Ztol:
            r=num_rounds
        else:
            r+=1

    ### TRAIN
    
    logging.info('Saving training examples...')
    print('Saving training examples...')
    with open(args.output+'/arcis_spec.p', 'wb') as file_arcis_spec:
        pickle.dump(arcis_spec, file_arcis_spec)
    with open(args.output+'/Y.p', 'wb') as file_np_theta:
        pickle.dump(np_theta, file_np_theta)
    
    tic = time()
    logging.info('Training SNPE...')
    print('Training SNPE...')
    if not reject:
        inference_object = inference.append_simulations(theta_aug, x, proposal=proposal)
        with open(args.output+'/inference.p', 'wb') as file_inference:
            pickle.dump(inference, file_inference)
            
    posterior_estimator = inference_object.train(show_train_summary=True, stop_after_epochs=args.patience, num_atoms=args.atoms, force_first_round_loss=True)
        
    if args.xnorm:
        default_x = xscaler.transform(obs_spec.reshape(1,-1))
    else:
        default_x = obs_spec.reshape(1,-1)

    ### GENERATE POSTERIOR
    
    print('\n Time elapsed: '+str(time()-tic))
    logging.info('Time elapsed: '+str(time()-tic))
    posterior = inference_object.build_posterior(sample_with='rejection').set_default_x(default_x)
    posteriors.append(posterior)
    print('Saving posteriors ')
    logging.info('Saving posteriors ')
    with open(args.output+'/posteriors.pt', 'wb') as file_posteriors:
        torch.save(posteriors, file_posteriors)
    proposal = posterior
        
    
    print('Saving samples ')
    logging.info('Saving samples ')
    tsamples = posterior.sample((5000,))
    if args.ynorm:
        samples.append(yscaler.inverse_transform(tsamples.cpu().detach().numpy()))
    else:
        samples.append(tsamples.cpu().detach().numpy())
        
    with open(args.output+'/samples.p', 'wb') as file_samples:
        pickle.dump(samples, file_samples)

    plt.close('all')

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

fig1 = corner(samples[-1], color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
with open(args.output+'corner_'+str(r+1)+'.jpg', 'wb') as file_post_equal_corner:
    plt.savefig(file_post_equal_corner, bbox_inches='tight')
plt.close('all')

print('\n')
print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
