#!/usr/bin/env python3

ARCiS = 'ARCiS'

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sklearn.decomposition import PCA
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE,SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
# import ultranest
from sbi.inference import SNPE_C, SNLE_A, SNRE_B
from time import time
import logging
import os
from tqdm import trange
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from corner import corner
from multiprocessing import Process, Pool
from simulator import simulator
from spectres import spectres
#import pywhatkit

supertic = time()

### PARSE COMMAND LINE ARGUMENTS ###

def parse_args():
    parser = argparse.ArgumentParser(description=('Train SNPE_C'))
    parser.add_argument('-input', type=str, help='ARCiS input file for the retrieval')
    parser.add_argument('-output', type=str, default='output/', help='Directory to save output')
    parser.add_argument('-model', type=str, default='nsf', help='Either nsf or maf.')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: CPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    parser.add_argument('-samples_per_round', type=str, default='1000', help='Number of samples to draw for training each round. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=50)
    parser.add_argument('-do_pca', action='store_true')
    parser.add_argument('-n_pca', type=int, default=50)
    parser.add_argument('-transforms', type=int, default=5)
    parser.add_argument('-bins', type=int, default=5)
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-embed_size', type=int, default=64)
    parser.add_argument('-embedding', action='store_true')
    parser.add_argument('-ynorm', action='store_true')
    parser.add_argument('-xnorm', action='store_true')
    parser.add_argument('-Ztol', type=float, default=0.5)
    parser.add_argument('-discard_prior_samples', action='store_true')
    parser.add_argument('-combined', action='store_true')
    parser.add_argument('-naug', type=int, default=5)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-retrain_from_scratch', action='store_true')
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-atoms', type=int, default=10)
    parser.add_argument('-method', type=str, default='snpe')
    parser.add_argument('-sample_with', type=str, default='rejection')
    parser.add_argument('-reuse_prior_samples', action='store_true')
    parser.add_argument('-samples_dir', type=str)
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
    logZ = np.median(-(P-pi-L).detach().numpy())
    logZp1 = np.percentile(-(P-pi-L).detach().numpy(), 84)
    logZm1 = np.percentile(-(P-pi-L).detach().numpy(), 16)
    return logZ, logZp1, logZm1

### Embedding network
class SummaryNet(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()
        inter = int((size_in+size_out)/2)
        self.fc1 = nn.Linear(size_in, inter)
        self.fc2 = nn.Linear(inter, size_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
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

embedding_net = SummaryNet(obs_spec.shape[0], args.embed_size)

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

if args.model == 'nsf':
    if args.embedding:
        neural_posterior = posterior_nn(model='nsf', hidden_features=args.hidden, num_transforms=args.transforms, num_bins=args.bins, num_blocks=args.blocks, embedding_net=embedding_net)
    else:
        neural_posterior = posterior_nn(model='nsf', hidden_features=args.hidden, num_transforms=args.transforms, num_bins=args.bins, num_blocks=args.blocks)    
else:
    neural_posterior = utils.posterior_nn(model=args.model)

if args.method=='snpe':
    inference = SNPE_C(prior = prior, density_estimator=neural_posterior, device=device)
elif args.method=='snle':
    inference = SNLE_A(prior = prior, density_estimator=neural_posterior, device=device)
elif args.method=='snre':
    inference = SNRE_B(prior = prior, density_estimator=neural_posterior, device=device)

nsamples = args.samples_per_round

try:
    samples_per_round = int(nsamples)
except:
    samples_per_round = nsamples.split()
    for i in range(len(samples_per_round)):
        samples_per_round[i] = int(samples_per_round[i])

try:
    num_rounds=len(samples_per_round)
except:
    num_rounds=num_rounds
    samples_per_round = num_rounds*[samples_per_round]
    
print('Training multi-round inference')
logging.info('Training multi-round inference')

proposal=prior
posteriors=[]

logZs = []
logZp1s = []
logZm1s = []

for r in range(num_rounds):
    print('\n')
    print('\n **** Training round ', r)
    logging.info('Round '+str(r))        
    
    if args.reuse_prior_samples and r==0:
        print('Reusing '+str(samples_per_round[0])+' prior samples from '+ args.samples_dir)
        logging.info('Reusing '+str(samples_per_round[0])+' prior samples from '+ args.samples_dir)
        arcis_spec = np.load(args.samples_dir+'/arcis_spec_round_'+str(0)+'.npy')[:samples_per_round[0]]
        np_theta = np.load(args.samples_dir+'/Y_round_'+str(0)+'.npy')[:samples_per_round[0]]
    else:
        ##### drawing samples and computing fwd models
        logging.info('Drawing '+str(samples_per_round[r])+' samples')
        print('Samples per round: ', samples_per_round[r])
        theta = proposal.sample((samples_per_round[r],))
        np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
                
        if args.ynorm:
            post_plot = yscaler.inverse_transform(np_theta)
        else:
            post_plot = np_theta
        
        fig1 = corner(post_plot, color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
        with open(args.output+'corner_'+str(r)+'.jpg', 'wb') as file_corner:
            plt.savefig(file_corner, bbox_inches='tight')
        plt.close('all')
                
        # COMPUTE MODELS
        
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
        
        arcis_spec = compute(np_theta)
        
        for j in range(args.processes):
            os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
                
        sm = np.sum(arcis_spec, axis=1)

        arcis_spec = arcis_spec[sm!=0]
        
        while len(arcis_spec)<samples_per_round[r]:
            remain = samples_per_round[r]-len(arcis_spec)
            print('ARCiS crashed, computing remaining ' +str(remain)+' models.')
            logging.info('ARCiS crashed, computing remaining ' +str(remain)+' models.')
            
            theta[len(arcis_spec):] = proposal.sample((remain,))
            np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
                        
            arcis_spec_ac=compute(np_theta[len(arcis_spec):])
                
            for j in range(args.processes):
                os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
                            
            sm_ac = np.sum(arcis_spec_ac, axis=1)

            arcis_spec = np.concatenate((arcis_spec, arcis_spec_ac[sm_ac!=0]))
        
        with open(args.output+'/arcis_spec_round_'+str(r)+'.npy', 'wb') as file_arcis_spec:
            np.save(file_arcis_spec, arcis_spec)
        with open(args.output+'/Y_round_'+str(r)+'.npy', 'wb') as file_np_theta:
            np.save(file_np_theta, np_theta)
    if r>0:
        logZ, logZp1, logZm1 = evidence(posteriors[-1], prior, arcis_spec, np_theta, obs_spec, noise_spec)
        logZs.append(logZ)
        logZp1s.append(logZp1)
        logZm1s.append(logZm1)
        print('\n')
        print('ln (Z) = ', round(logZ, 2))
        print('\n')
    
    if r>1 and (logZs[-1]-logZs[-2]<args.Ztol) and (logZs[-2]-logZs[-3]<args.Ztol):
        num_rounds=r-1
        break
    
    theta = torch.tensor(np.repeat(np_theta, args.naug, axis=0), dtype=torch.float32, device=device)
    arcis_spec_aug = np.repeat(arcis_spec, args.naug, axis=0) + noise_spec*np.random.randn(samples_per_round[r]*args.naug, obs_spec.shape[0])
    
    if r==0:
        ## Fit PCA and xscaler with samples from prior only
        if args.do_pca:
            print('Fitting PCA...')
            logging.info('Fitting PCA...')
            pca = PCA(n_components=args.n_pca)
            pca.fit(arcis_spec)
            with open(args.output+'/pca.p', 'wb') as file_pca:
                pickle.dump(pca, file_pca)
            if args.xnorm:
                xscaler = StandardScaler().fit(pca.transform(arcis_spec))
                with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                    pickle.dump(xscaler, file_xscaler)
        elif args.xnorm:
            xscaler = StandardScaler().fit(arcis_spec)
            with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                pickle.dump(xscaler, file_xscaler)
    
    if args.do_pca:
        x_i = pca_trans.transform(arcis_spec_aug)
        if args.xnorm:
            x_f = xscaler.transform(x_i)
        else:
            x_f = x_i
    elif args.xnorm:
        x_f = xscaler.transform(arcis_spec_aug)
    else:
        x_f = arcis_spec_aug
            
    x = torch.tensor(x_f, dtype=torch.float32, device=device)
        
    logging.info('Training...')
    tic = time()
    if args.method=='snpe':
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(
            discard_prior_samples=args.discard_prior_samples, use_combined_loss=args.combined, show_train_summary=True, 
            stop_after_epochs=args.patience, num_atoms=args.atoms, retrain_from_scratch=False, 
            force_first_round_loss=True)
    elif args.method=='snle':
        density_estimator = inference.append_simulations(theta, x).train(
            discard_prior_samples=args.discard_prior_samples, show_train_summary=True, 
            stop_after_epochs=args.patience)

    print('\n Time elapsed: '+str(time()-tic))
    logging.info('Time elapsed: '+str(time()-tic))
    try:
        posterior = inference.build_posterior(density_estimator, sample_with=args.sample_with)
    except:
        print('\n OH NO!, IT HAPPENED!')
        logging.info('OH NO!, IT HAPPENED!')
        try:
            posterior = inference.build_posterior(density_estimator, sample_with=args.sample_with)
        except:
            print('\n OH NO!, IT HAPPENED *AGAIN*!?')
            logging.info('OH NO!, IT HAPPENED *AGAIN*!?')
            posterior = inference.build_posterior(density_estimator, sample_with=args.sample_with)
    posteriors.append(posterior)
    print('Saving posteriors ')
    logging.info('Saving posteriors ')
    with open(args.output+'/posteriors.pt', 'wb') as file_posteriors:
        torch.save(posteriors, file_posteriors)
    
    if args.do_pca:
        default_x_pca = pca.transform(obs_spec.reshape(1,-1))
        if args.xnorm:
            default_x = xscaler.transform(default_x_pca)
        else:
            default_x = default_x_pca
    elif args.xnorm:
        default_x = xscaler.transform(obs_spec.reshape(1,-1))
    else:
        default_x = obs_spec.reshape(1,-1)
                        
    proposal = posterior.set_default_x(default_x)
            
    plt.close('all')

print('Drawing samples ')
logging.info('Drawing samples ')
samples = []
for j in range(num_rounds):
    print('Drawing samples from round ', j)
    logging.info('Drawing samples from round ' + str(j))
    posterior = posteriors[j]
    tsamples = posterior.sample((10000,), x=default_x, show_progress_bars=True)
    if args.ynorm:
        samples.append(yscaler.inverse_transform(tsamples.cpu().detach().numpy()))
    else:
        samples.append(tsamples.cpu().detach().numpy())

# Is this actually necessary??
print('Saving samples ')
logging.info('Saving samples ')
with open(args.output+'/samples.p', 'wb') as file_samples:
    pickle.dump(samples, file_samples)

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

fig1 = corner(samples[-1], color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
with open(args.output+'corner_'+str(r+1)+'.jpg', 'wb') as file_post_equal_corner:
    plt.savefig(file_post_equal_corner, bbox_inches='tight')
plt.close('all')

print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
