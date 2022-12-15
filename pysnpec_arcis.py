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
    parser.add_argument('-obs_trans', type=str, default='None', help='File with transit observation')
    parser.add_argument('-obs_phase', type=str, default='None', help='File with phase curve observation')
    parser.add_argument('-obs_emis', type=str, default='None', help='File with eclipse observation')
    parser.add_argument('-output', type=str, default='output/', help='Directory to save output')
    parser.add_argument('-prior', type=str, default='prior.dat', help='File with prior bounds (box uniform)')
    parser.add_argument('-model', type=str, default='nsf', help='Either nsf or maf.')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: GPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    parser.add_argument('-samples_per_round', type=str, default='5000', help='Number of samples to draw for training each round. If a single value, that number of samples will be drawn for num_rounds rounds. If multiple values, at each round, the number of samples specified will be drawn. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=50)
    parser.add_argument('-do_pca', action='store_true')
    parser.add_argument('-n_pca_trans', type=int, default=50)
    parser.add_argument('-n_pca_phase', type=int, default=50)
    parser.add_argument('-transforms', type=int, default=5)
    parser.add_argument('-bins', type=int, default=5)
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-embed_size', type=int, default=64)
    parser.add_argument('-embedding', action='store_true')
    parser.add_argument('-ynorm', action='store_true')
    parser.add_argument('-xnorm', action='store_true')
    parser.add_argument('-discard_prior_samples', action='store_true')
    parser.add_argument('-combined', action='store_true')
    parser.add_argument('-naug', type=int, default=5)
    parser.add_argument('-clean', action='store_true')
    parser.add_argument('-dont_plot', action='store_false')
    parser.add_argument('-removeR', action='store_true')
    parser.add_argument('-resume', action='store_false')
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-scaleR', action='store_true')
    parser.add_argument('-retrain_from_scratch', action='store_true')
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-atoms', type=int, default=10)
    parser.add_argument('-method', type=str, default='snpe')
    parser.add_argument('-sample_with', type=str, default='rejection')
    parser.add_argument('-reuse_prior_samples', action='store_true')
    parser.add_argument('-samples_dir', type=str)
    return parser.parse_args()

### Embedding network
class SummaryNet(nn.Module): 

    def __init__(self, size_in, size_out): 
        super().__init__()
        self.fc1 = nn.Linear(size_in, 128)
        # self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, size_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
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

### 1. Load observation/s
print('Loading observations... ')
logging.info('Loading observations...')
if args.obs_trans!='None':
    wvl_trans = np.loadtxt(args.obs_trans)[:,0]
    obs_trans = np.loadtxt(args.obs_trans)[:,1]
    noise_trans = np.loadtxt(args.obs_trans)[:,2]
if args.obs_phase!='None':
    print('Reading phase curve...')
    logging.info('Reading phase curve...')
    phase_str = (args.obs_phase).split()
    nwvl = np.zeros(len(phase_str))
    for i in range(len(phase_str)):
        nwvl[i] = len(np.loadtxt(phase_str[i])[:,0])
    l=[0]
    obs_phase = np.zeros(int(sum(nwvl)))
    noise_phase =np.zeros(int(sum(nwvl)))
    for j in range(len(phase_str)):
        phasej = np.loadtxt(phase_str[j])
        l.append(len(phasej))
        obs_phase[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,1]
        noise_phase[sum(l[:j+1]):sum(l[:j+2])] = phasej[:,2]  #### obs_phase 1D array of flattened phasecurves (len = sum of lens of each phase observation)
if args.obs_emis!='None':
    obs_emis = np.loadtxt(args.obs_emis)[:,1]
    noise_emis = np.loadtxt(args.obs_emis)[:,2]

#embedding_net = SummaryNet(obs_trans.shape[0], args.embed_size)

### 2. Load, create, and transform prior
print('Loading prior from '+args.prior)
logging.info('Loading prior from '+args.prior)
    
prior_bounds = np.loadtxt(args.prior)

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

############
### Resuming
if args.resume:
    posteriors = []                         
    proposal = prior
else:
    posteriors = torch.load(args.output+'/posteriors.pt')
    if args.xnorm and not args.do_pca:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        if args.obs_phase!='None':
            default_x = xscaler.transform(np.concatenate((obs_trans.reshape(1,-1),
                                obs_phase.reshape(1,-1)), axis=1))
        proposal = posteriors[-1].set_default_x(default_x)
    elif args.do_pca and not args.xnorm:
        pca = pickle.load(open(args.output+'/pca.p', 'rb'))
        proposal = posteriors[-1].set_default_x(pca.transform(x_o[:,1].reshape(1,-1)))
    elif args.do_pca and args.xnorm:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        pca_trans = pickle.load(open(args.output+'/pca_trans.p', 'rb'))
        if args.obs_phase!='None':
            pca_phase = pickle.load(open(args.output+'/pca_phase.p', 'rb'))
            obs_phase_flat = np.zeros(int(sum(nwvl)))
            for i in range(10):
                new_wvl = obs_phase[i][:,0]
                obs_phase_flat[int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = obs_phase[i][:,1]
            
            default_x_pca = np.concatenate((pca_trans.transform(obs_trans.reshape(1,-1)),
                                pca_phase.transform(obs_phase.reshape(1,-1))), axis=1)
        else:
            default_x_pca = pca_trans.transform(obs_trans.reshape(1,-1))
        default_x = xscaler.transform(default_x_pca)
        proposal = posteriors[-1].set_default_x(default_x)
    else:
        proposal = posteriors[-1].set_default_x(x_o[:,1])
#######

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

for r in range(len(posteriors), num_rounds):
    print('\n')
    print('\n **** Training round ', r)
    logging.info('Round '+str(r))
    
    if args.reuse_prior_samples and r==0:
        print('Reusing '+str(samples_per_round[0])+' prior samples from '+ args.samples_dir)
        logging.info('Reusing '+str(samples_per_round[0])+' prior samples from '+ args.samples_dir)
        trans = np.load(args.samples_dir+'/trans_round_'+str(0)+'.npy')[:samples_per_round[0]]
        if args.obs_phase!='None':
            phase = np.load(args.samples_dir+'/phase_round_'+str(0)+'.npy')[:samples_per_round[0]]#########
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
        
        fig1 = corner(post_plot, color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds)
        with open(args.output+'corner_'+str(r)+'.jpg', 'wb') as file_corner:
            plt.savefig(file_corner, bbox_inches='tight')
        plt.close('all')
        
#        pywhatkit.sendwhats_image("+31613945493", args.output+'corner_'+str(r)+'.jpg', 'Round '+str(r), 32, True)
        
        # COMPUTE MODELS
        
        def compute(np_theta):
            samples_per_process = len(np_theta)//args.processes

            print('Samples per process: ', samples_per_process)

            parargs=[]
            if args.ynorm:
                params=yscaler.inverse_transform(np_theta)
            else:
                params = np_theta
                            
            for i in range(args.processes-1):
                parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, i))
            parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, args.processes-1))

            tic=time()
            pool = Pool(processes = args.processes)
            if args.obs_phase!='None':
                trans_phase = pool.starmap(simulator, parargs)
                trans = np.zeros([len(np_theta), len(obs_trans)])
                phase = np.zeros([len(np_theta), int(sum(nwvl))])
                print('Joining processes...')
                for i in trange(args.processes-1):
                    trans[i*samples_per_process:(i+1)*samples_per_process] = trans_phase[i][0]
                    phase[i*samples_per_process:(i+1)*samples_per_process] = trans_phase[i][1]
                trans[(args.processes-1)*samples_per_process:] = trans_phase[(args.processes-1)][0]
                phase[(args.processes-1)*samples_per_process:] = trans_phase[(args.processes-1)][1]
                print('Time elapsed: ', time()-tic)
                logging.info(('Time elapsed: ', time()-tic))
                return trans, phase
            else:
                trans_s = pool.starmap(simulator, parargs)
                trans = np.concatenate(trans_s)
                print('Time elapsed: ', time()-tic)
                logging.info(('Time elapsed: ', time()-tic))
                return trans
        
        if args.obs_phase!='None':
            trans,phase = compute(np_theta)
        else:
            trans = compute(np_theta)
        
        if args.clean:
            for j in range(args.processes):
                os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
                
        sm = np.sum(trans, axis=1)

        trans = trans[sm!=0]
        if args.obs_phase!='None':
            phase = phase[sm!=0]
        
        while len(trans)<samples_per_round[r]:
            remain = samples_per_round[r]-len(trans)
            print('ARCiS crashed, computing remaining ' +str(remain)+' models.')
            logging.info('ARCiS crashed, computing remaining ' +str(remain)+' models.')
            
            theta[len(trans):] = proposal.sample((remain,))
            np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
                        
            if args.obs_phase!='None':
                trans_ac,phase_ac=compute(np_theta[len(trans):])
            else:
                trans_ac=compute(np_theta[len(trans):])
                
            if args.clean:
                for j in range(args.processes):
                    os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
                            
            sm_ac = np.sum(trans_ac, axis=1)

            trans = np.concatenate((trans, trans_ac[sm_ac!=0]))
            if args.obs_phase!='None':
                phase = np.concatenate((phase, phase_ac[sm_ac!=0]))
        
        with open(args.output+'/trans_round_'+str(r)+'.npy', 'wb') as file_trans:
            np.save(file_trans, trans)
        if args.obs_phase!='None':
            with open(args.output+'/phase_round_'+str(r)+'.npy', 'wb') as file_phase:
                np.save(file_phase, phase)
        with open(args.output+'/Y_round_'+str(r)+'.npy', 'wb') as file_np_theta:
            np.save(file_np_theta, np_theta)
                
        plt.figure(figsize=(15,5))
        plt.errorbar(x = wvl_trans, y=obs_trans, yerr=noise_trans, color='red', ls='', fmt='.', label='Observation')
        plt.plot(wvl_trans, np.median(trans, axis=0), c='mediumblue', label='Round '+str(r)+' fit')
        plt.fill_between(wvl_trans, np.percentile(trans, 84, axis=0), np.percentile(trans, 16, axis=0), color='mediumblue', alpha=0.4)
        plt.fill_between(wvl_trans, np.percentile(trans, 97.8, axis=0), np.percentile(trans, 2.2, axis=0), color='mediumblue', alpha=0.1)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Transit depth')
        plt.legend()
        with open(args.output+'round_'+str(r)+'_trans.jpg', 'wb') as file_trans_fit:
            plt.savefig(file_trans_fit, bbox_inches='tight')
        plt.close('all')
#        pywhatkit.sendwhats_image("+31613945493", args.output+'round_'+str(r)+'_trans.jpg', 'Round '+str(r), 32, True)
        
        if args.obs_phase!='None':
            plt.figure(figsize=(15,5))
            plt.errorbar(x = np.arange(0,int(sum(nwvl))), y=obs_phase, yerr=noise_phase, color='red', ls='', fmt='.', label='Observation')
            plt.plot(np.arange(0,int(sum(nwvl))), np.median(phase, axis=0), c='mediumblue', label='Round '+str(r)+' fit')
            plt.fill_between(np.arange(0,int(sum(nwvl))), np.percentile(phase, 84, axis=0), np.percentile(phase, 16, axis=0), color='mediumblue', alpha=0.4)
            plt.fill_between(np.arange(0,int(sum(nwvl))), np.percentile(phase, 97.8, axis=0), np.percentile(phase, 2.2, axis=0), color='mediumblue', alpha=0.1)
            plt.ylabel('Transit depth')
            plt.legend()
            with open(args.output+'round_'+str(r)+'_phase.jpg', 'wb') as file_phase_fit:
                plt.savefig(file_phase_fit, bbox_inches='tight')
            plt.close('all')
#            pywhatkit.sendwhats_image("+31613945493", args.output+'round_'+str(r)+'_phase.jpg', 'Round '+str(r), 32, True)
#     #######
     
    theta = torch.tensor(np.repeat(np_theta, args.naug, axis=0), dtype=torch.float32, device=device)
    trans_aug = np.repeat(trans, args.naug, axis=0) + noise_trans*np.random.randn(samples_per_round[r]*args.naug, obs_trans.shape[0])
    if args.obs_phase!='None':
        phase_aug = np.repeat(phase, args.naug, axis=0) + noise_phase*np.random.randn(samples_per_round[r]*args.naug, phase.shape[1])
    
    if r==0:
        ## Fit PCA and xscaler with samples from prior only
        if args.do_pca:
            print('Fitting PCA...')
            logging.info('Fitting PCA...')
            pca_trans = PCA(n_components=args.n_pca_trans)
            pca_trans.fit(trans)
            with open(args.output+'/pca_trans.p', 'wb') as file_pca_trans:
                pickle.dump(pca_trans, file_pca_trans)
            if args.obs_phase!='None':
                pca_phase = PCA(n_components=args.n_pca_phase)
                pca_phase.fit(phase)
                with open(args.output+'/pca_phase.p', 'wb') as file_pca_phase:
                    pickle.dump(pca_phase, file_pca_phase)
            if args.xnorm:
                if args.obs_phase!='None':
                    xscaler = StandardScaler().fit(np.concatenate((pca_trans.transform(trans_aug), 
                                                                   pca_phase.transform(phase_aug)), axis=1))
                else:
                    xscaler = StandardScaler().fit(pca_trans.transform(trans))
                with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                    pickle.dump(xscaler, file_xscaler)
        elif args.xnorm:
            if args.obs_phase!='None':
                xscaler = StandardScaler().fit(np.concatenate((trans, phase), axis=1))
            else:
                xscaler = StandardScaler().fit(trans)
            with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                pickle.dump(xscaler, file_xscaler)
    
    if args.do_pca:
        if args.obs_phase!='None':
            x_i = np.concatenate((pca_trans.transform(trans_aug), pca_phase.transform(phase_aug)), axis=1)
        else:
            x_i = pca_trans.transform(trans_aug)
        if args.xnorm:
            x_f = xscaler.transform(x_i)
        else:
            x_f = x_i
    elif args.xnorm:
        print('Not doing any PCA...')
        logging.info('Not doing any PCA...')
        if args.obs_phase!='None':
            x_f = xscaler.transform(np.concatenate((trans_aug, phase_aug), axis=1))
        else:
            x_f = xscaler.transform(trans_aug)
    else:
        if args.obs_phase!='None':
            x_f = np.concatenate((trans_aug, phase_aug), axis=1)
        else:
            x_f = trans_aug
            
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
        if args.obs_phase!='None':
            default_x_pca = np.concatenate((pca_trans.transform(obs_trans.reshape(1,-1)),
                            pca_phase.transform(obs_phase.reshape(1,-1))), axis=1)
        else:
            default_x_pca = pca_trans.transform(obs_trans.reshape(1,-1))
        if args.xnorm:
            default_x = xscaler.transform(default_x_pca)
        else:
            default_x = default_x_pca
    elif args.xnorm:
        if args.obs_phase!='None':
            default_x = xscaler.transform(np.concatenate((obs_trans.reshape(1,-1), obs_phase.reshape(1,-1)), axis=1))
        else:
            default_x = xscaler.transform(obs_trans.reshape(1,-1))
    else:
        if args.obs_phase!='None':
            default_x = np.concatenate((obs_trans.reshape(1,-1), obs_phase.reshape(1,-1)), axis=1)
        else:
            default_x = obs_trans.reshape(1,-1)
                        
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

print('Saving samples ')
logging.info('Saving samples ')
with open(args.output+'/samples.p', 'wb') as file_samples:
    pickle.dump(samples, file_samples)

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

#if args.dont_plot:
fig1 = corner(samples[-1], smooth=0.5, range=prior_bounds)
with open(args.output+'corner_'+str(num_rounds)+'.jpg', 'wb') as file_post_equal_corner:
    plt.savefig(file_post_equal_corner, bbox_inches='tight')
plt.close('all')
#pywhatkit.sendwhats_image("+31613945493", args.output+'corner_'+str(num_rounds)+'.jpg', 'Final corner plot', 32, True)

print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
