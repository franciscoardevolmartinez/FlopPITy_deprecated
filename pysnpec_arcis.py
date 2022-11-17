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

supertic = time()

### PARSE COMMAND LINE ARGUMENTS ###

def parse_args():
    parser = argparse.ArgumentParser(description=('Train SNPE_C'))
    parser.add_argument('-input', type=str, help='ARCiS input file for the retrieval')
    parser.add_argument('-obs_trans', type=str, default='None', help='File with transit observation')
    parser.add_argument('-obs_phase', type=str, default='None', help='File with phase curve observation')
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
    
def find_repeat(array):
    repeat = np.empty(len(array), dtype=bool)
    repeat[0]=True
    for i in range(1,len(array)):
        if array[i]==array[i-1]:
            repeat[i]=False
        else:
            repeat[i]=True
    return repeat
    
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
    obs_trans=np.loadtxt(args.obs_trans)
if args.obs_phase!='None':
    phase = (args.obs_phase).split()
    obs_phase=[]
    for i in range(len(phase)):
        obs_phase.append(np.loadtxt(phase[i]))

embedding_net = SummaryNet(obs_trans.shape[0], args.embed_size)

### 2. Load, create, and transform prior
print('Loading prior from '+args.prior)
logging.info('Loading prior from '+args.prior)
    
prior_bounds = np.loadtxt(args.prior)

if args.ynorm:
    yscaler = Normalizer(prior_bounds)
    pickle.dump(yscaler, open(args.output+'/yscaler.p', 'wb'))

    prior_min = torch.tensor(yscaler.transform(prior_bounds[:,0].reshape(1, -1)).reshape(-1))
    prior_max = torch.tensor(yscaler.transform(prior_bounds[:,1].reshape(1, -1)).reshape(-1))
else:
    prior_min = torch.tensor(prior_bounds[:,0].reshape(1,-1))
    prior_max = torch.tensor(prior_bounds[:,1].reshape(1,-1))

prior = utils.BoxUniform(low=prior_min.to(device, non_blocking=True), high=prior_max.to(device, non_blocking=True), device=device)

num_rounds = args.num_rounds

if args.resume:
    posteriors = []                         
    proposal = prior
else:
    posteriors = torch.load(args.output+'/posteriors.pt')
    if args.xnorm and not args.do_pca:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        if args.obs_phase!='None':
            nwvl = np.zeros(len(phase))
            for i in range(len(phase)):
                nwvl[i] = len(obs_phase[i][:,0])
            obs_phase_flat = np.zeros(int(sum(nwvl)))
            for i in range(10):
                new_wvl = obs_phase[i][:,0]
                obs_phase_flat[int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = obs_phase[i][:,1]
            
            default_x = xscaler.transform(np.concatenate((obs_trans[:,1].reshape(1,-1), 
                                obs_phase_flat.reshape(1,-1)), axis=1))
        proposal = posteriors[-1].set_default_x(default_x)
    elif args.do_pca and not args.xnorm:
        pca = pickle.load(open(args.output+'/pca.p', 'rb'))
        proposal = posteriors[-1].set_default_x(pca.transform(x_o[:,1].reshape(1,-1)))
    elif args.do_pca and args.xnorm:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        pca_trans = pickle.load(open(args.output+'/pca_trans.p', 'rb'))
        if args.obs_phase!='None':
            pca_phase = pickle.load(open(args.output+'/pca_phase.p', 'rb'))
        
            nwvl = np.zeros(len(phase))
            for i in range(len(phase)):
                nwvl[i] = len(obs_phase[i][:,0])
            obs_phase_flat = np.zeros(int(sum(nwvl)))
            for i in range(10):
                new_wvl = obs_phase[i][:,0]
                obs_phase_flat[int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = obs_phase[i][:,1]
            
            default_x_pca = np.concatenate((pca_trans.transform(obs_trans[:,1].reshape(1,-1)), 
                                pca_phase.transform(obs_phase_flat.reshape(1,-1))), axis=1)
        else:
            default_x_pca = pca_trans.transform(obs_trans[:,1].reshape(1,-1))
        default_x = xscaler.transform(default_x_pca)
        proposal = posteriors[-1].set_default_x(default_x)
    else:
        proposal = posteriors[-1].set_default_x(x_o[:,1])

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

pattern=[False, False, True]
retrain = (num_rounds//3)*pattern+pattern[:num_rounds-3*(num_rounds//3)]

for r in range(len(posteriors), num_rounds):
    print('\n')
    print('\n **** Training round ', r)
    logging.info('Round '+str(r))
    
    # if args.reuse_prior_samples and r==0:
    #     print('Reusing prior samples')
    #     logging.info('Reusing prior samples')
    #     X = np.load(args.samples_dir+'/X_round_'+str(r)+'.npy')[:samples_per_round[r]]
    #     if args.phasecurve:
    #         phase = np.load(args.samples_dir+'/phase_round_'+str(r)+'.npy')[:samples_per_round[r]]#########
    #     np_theta = np.load(args.samples_dir+'/Y_round_'+str(r)+'.npy')[:samples_per_round[r]]
    # else:
    logging.info('Drawing '+str(samples_per_round[r])+' samples')
    print('Samples per round: ', samples_per_round[r])
    theta = proposal.sample((samples_per_round[r],))
    np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
    
    if args.dont_plot:
        if args.ynorm:
            post_plot = yscaler.inverse_transform(np_theta)
        else:
            post_plot = np_theta
        fig1 = corner(post_plot, smooth=0.5, range=prior_bounds)
        plt.savefig(args.output+'corner_'+str(r)+'.pdf', bbox_inches='tight')
        
    
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
            wvl_trans = np.loadtxt(args.output+'round_'+str(r)+str(0)+'_out/model000001/trans')[:,0]
            wvl_phase = np.loadtxt(args.output+'round_'+str(r)+str(0)+'_out/model000001/phase')[:,0]
            trans = np.zeros([len(np_theta), len(wvl_trans)])
            phase = np.zeros([len(np_theta), len(obs_phase), len(wvl_phase)])
            print('Joining processes...')
            for i in trange(args.processes-1):
                trans[i*samples_per_process:(i+1)*samples_per_process] = trans_phase[i][0]
                phase[i*samples_per_process:(i+1)*samples_per_process] = trans_phase[i][1]
            trans[(args.processes-1)*samples_per_process:] = trans_phase[(args.processes-1)][0]
            phase[(args.processes-1)*samples_per_process:] = trans_phase[(args.processes-1)][1]
            print('Time elapsed: ', time()-tic)
            logging.info(('Time elapsed: ', time()-tic))
            print('*Compute* trans shape: ', trans.shape)
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
        
    
    # if args.obs_phase!='None':
    #     trans, phase = simulator(params, args.output, r, args.input, 0)
    # else:
    #     trans = simulator(params, args.output, r, args.input, 0)
        
    dirx = args.output + 'round_'+str(r)+str(0)+'_out/'
    wvl_trans = np.loadtxt(args.output+'round_'+str(r)+str(0)+'_out/model000001/trans')[:,0]
    if args.obs_phase!='None':
        wvl_phase = np.loadtxt(args.output+'round_'+str(r)+str(0)+'_out/model000001/phase')[:,0]
    if args.clean:
        for j in range(args.processes):
            os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
            
    sm = np.sum(trans, axis=1)

    trans = trans[sm!=0]
    if args.obs_phase!='None':
        phase = phase[sm!=0]
    
    print('Trans shape: ', trans.shape)
    
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

#         if args.ynorm:
#             params = yscaler.inverse_transform(np_theta[len(trans):])
#         else:
#             params = np_theta[len(trans):]
            
#         for j in range(args.processes):
#                 os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')
        
#         tic=time()
#         if args.obs_phase!='None':
#             trans_ac, phase_ac = simulator(params, args.output, r, args.input, 0)
#         else:
#             trans_ac = simulator(params, args.output, r, args.input, 0)
#         print('Time elapsed: ', time()-tic)
#         logging.info(('Time elapsed: ', time()-tic))
        
#         print('Trans_ac', trans_ac.shape)
#         print('Phase_ac', phase_ac.shape)
        
        sm_ac = np.sum(trans_ac, axis=1)

        trans = np.concatenate((trans, trans_ac[sm_ac!=0]))
        if args.obs_phase!='None':
            phase = np.concatenate((phase, phase_ac[sm_ac!=0]))
            
        print('Trans', trans.shape)
        print('Phase', phase.shape)
            
        # np_theta = np.concatenate((np_theta, np_theta_ac[:len(trans_ac)]))
        
        # print(np_theta.shape)
                        
        
    trans_noise = obs_trans[:,2]
    
    print('Rebinning transmission spectrum...')
    wvl_obs = np.round(obs_trans[:,0],6)
    sel_ti = np.in1d(wvl_trans, wvl_obs)
    repeat = find_repeat(wvl_trans)
    sel_t = sel_ti*repeat
    trans_reb = trans[:,sel_t]
    
    if args.obs_phase!='None':
        print('Rebinning emission spectra...')
        logging.info('Rebinning emission spectra...')
        nwvl = np.zeros(phase.shape[1]) #count wvl bins in each phase 
        for i in range(phase.shape[1]):
            nwvl[i] = len(obs_phase[i][:,0])
            ## Rebin spectra
        phase_reb = np.zeros([phase.shape[0], int(sum(nwvl))])
        phase_noise = np.zeros(int(sum(nwvl)))
        obs_phase_flat = np.zeros(int(sum(nwvl)))
        for i in range(phase.shape[1]):
            new_wvl = obs_phase[i][:,0]
            phase_reb[:,int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = spectres(new_wvl, wvl_phase,
                                                                          phase[:,i,:]) #rebin model to observation wvl and put in a flat array with all other phases
            phase_noise[int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = obs_phase[i][:,2]
            obs_phase_flat[int(sum(nwvl[:i])):int(sum(nwvl[:i+1]))] = obs_phase[i][:,1] #flatten the observations
            
    np.save(args.output+'/trans_round_'+str(r)+'.npy', trans)
    if args.obs_phase!='None':
        np.save(args.output+'/phase_round_'+str(r)+'.npy', phase)
    np.save(args.output+'/Y_round_'+str(r)+'.npy', np_theta)
            
    if args.dont_plot:
        plt.figure(figsize=(15,5))
        plt.errorbar(x = x_o[:,0], y=x_o[:,1], yerr=x_o[:,2], color='red', ls='', fmt='.', label='Observation')
        plt.plot(x_o[:,0], np.median(X, axis=0), c='mediumblue', label='Round '+str(r)+' models')
        plt.fill_between(x_o[:,0], np.percentile(X, 84, axis=0), np.percentile(X, 16, axis=0), color='mediumblue', alpha=0.4)
        plt.fill_between(x_o[:,0], np.percentile(X, 97.8, axis=0), np.percentile(X, 2.2, axis=0), color='mediumblue', alpha=0.1)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel('Transit depth')
        plt.legend()
        plt.savefig(args.output+'round_'+str(r)+'_trans.pdf', bbox_inches='tight')
        
    theta = torch.tensor(np.repeat(np_theta, args.naug, axis=0), dtype=torch.float32, device=device)
    trans_aug = np.repeat(trans_reb, args.naug, axis=0) + trans_noise*np.random.randn(samples_per_round[r]*args.naug,
                                                                                  obs_trans.shape[0])
    if args.obs_phase!='None':
        phase_reb_aug = np.repeat(phase_reb, args.naug, axis=0) + phase_noise*np.random.randn(samples_per_round[r]*args.naug,
                                                                                  phase_reb.shape[1])
    ### ZOOM ON HST
    plt.figure(figsize=(25, 20))
    plt.plot(new_wvl[0:15], phase_reb[0, 0:15], 'd-', c='limegreen', label='Rebinned')
    plt.plot(wvl_phase[0:15], phase[0, 0, 0:15], 'o-', c='red', label='Original model')
    for i in range(naug):
        plt.plot(new_wvl[0:15], phase_reb_aug[i, 0:15], c='lightskyblue', label='Noisy')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Relative flux')
    plt.legend()
    plt.savefig(args.output+'spectra_hst.pdf', bbox_inches='tight')
    
    ### INCLUDING SPITZER
    plt.figure(figsize=(25, 20))
    plt.plot(new_wvl[0:17], phase_reb[0, 0:17], 'd-', c='limegreen', label='Rebinned')
    plt.plot(wvl_phase[0:17], phase[0, 0, 0:17], 'o-', c='red', label='Original model')
    for i in range(naug):
        plt.plot(new_wvl[0:17], phase_reb_aug[i, 0:17], c='lightskyblue', label='Noisy')
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel('Relative flux')
    plt.legend()
    plt.savefig(args.output+'spectra_spitzer.pdf', bbox_inches='tight')
    
    if r==0:
        ## Fit PCA and xscaler with samples from prior only
        if args.do_pca:
            pca_trans = PCA(n_components=args.n_pca_trans)
            pca_trans.fit(trans_reb)
            pickle.dump(pca_trans, open(args.output+'/pca_trans.p', 'wb'))
            if args.obs_phase!='None':
                pca_phase = PCA(n_components=args.n_pca_phase)
                pca_phase.fit(phase_reb)
                pickle.dump(pca_phase, open(args.output+'/pca_phase.p', 'wb'))
            if args.xnorm:
                if args.obs_phase!='None':
                    xscaler = StandardScaler().fit(np.concatenate((pca_trans.transform(trans_aug), 
                                                                   pca_phase.transform(phase_reb_aug)), axis=1))
                else:
                    xscaler = StandardScaler().fit(pca_trans.transform(trans_reb))
                pickle.dump(xscaler, open(args.output+'/xscaler.p', 'wb'))
        elif args.xnorm:
            if args.obs_phase!='None':
                xscaler = StandardScaler().fit(np.concatenate((trans_reb, phase_reb), axis=1))
            else:
                xscaler = StandardScaler().fit(trans_reb)
            pickle.dump(xscaler, open(args.output+'/xscaler.p', 'wb'))
    
    if args.do_pca:
        if args.obs_phase!='None':
            x_i = np.concatenate((pca_trans.transform(trans_aug), pca_phase.transform(phase_reb_aug)), axis=1)
        else:
            x_i = pca_trans.transform(trans_aug)
        if args.xnorm:
            x_f = xscaler.transform(x_i)
        else:
            x_f = x_i
    elif args.xnorm:
        if args.obs_phase!='None':
            x_f = xscaler.transform(np.concatenate((trans_aug, phase_reb_aug), axis=1))
        else:
            x_f = xscaler.transform(trans_aug)
    else:
        if args.obs_phase!='None':
            x_f = np.concatenate((trans_aug, phase_reb_aug), axis=1)
        else:
            x_f = trans_aug
            
    x = torch.tensor(x_f, dtype=torch.float32, device=device)
    
    plt.figure(figsize=(15,5))
    for i in range(len(x_f)):
        plt.plot(x_f[i], 'b', alpha=0.5)
    plt.savefig(args.output+'round_'+str(r)+'_trans.pdf', bbox_inches='tight')
    
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
    torch.save(posteriors, args.output+'/posteriors.pt')
    
    if args.do_pca:
        if args.obs_phase!='None':
            default_x_pca = np.concatenate((pca_trans.transform(obs_trans[:,1].reshape(1,-1)), 
                            pca_phase.transform(obs_phase_flat.reshape(1,-1))), axis=1)
        else:
            default_x_pca = pca_trans.transform(obs_trans[:,1].reshape(1,-1))
        if args.xnorm:
            default_x = xscaler.transform(default_x_pca)
        else:
            default_x = default_x_pca
    elif args.xnorm:
        if args.obs_phase!='None':
            default_x = xscaler.transform(np.concatenate((obs_trans[:,1].reshape(1,-1), obs_phase_flat.reshape(1,-1)), axis=1))
        else:
            default_x = xscaler.transform(obs_trans[:,1].reshape(1,-1))
    else:
        if args.obs_phase!='None':
            default_x = np.concatenate((obs_trans[:,1].reshape(1,-1), obs_phase_flat.reshape(1,-1)), axis=1)
        else:
            default_x = obs_trans[:,1].reshape(1,-1)
            
    proposal = posterior.set_default_x(default_x)
            
    plt.close('all')

print('Drawing samples ')
logging.info('Drawing samples ')
samples = []
for j in range(num_rounds):
    print('Drawing samples from round ', j)
    logging.info('Drawing samples from round ' + str(j))
    # if args.xnorm and not args.do_pca:
    #     obs = torch.tensor(xscaler.transform(x_o[:,1].reshape(1,-1)), device=device)
    # elif args.do_pca and not args.xnorm:
    #     obs = torch.tensor(pca.transform(x_o[:,1].reshape(1,-1)), device=device)
    # elif args.do_pca and args.xnorm:
    #     obs = torch.tensor(xscaler.transform(pca.transform(x_o[:,1].reshape(1,-1))), device=device)
    # else:
    #     obs = torch.tensor(x_o[:,1].reshape(1,-1), device=device)
    posterior = posteriors[j]
    tsamples = posterior.sample((2000,), x=default_x, show_progress_bars=True)
    if args.ynorm:
        samples.append(yscaler.inverse_transform(tsamples.cpu().detach().numpy()))
    else:
        samples.append(tsamples.cpu().detach().numpy())

print('Saving samples ')
logging.info('Saving samples ')
pickle.dump(samples, open(args.output+'/samples.p', 'wb'))

np.savetxt(args.output+'/post_equal_weights.txt', samples[-1])

if args.dont_plot:
    fig1 = corner(samples[-1], smooth=0.5, range=prior_bounds)
    plt.savefig(args.output+'corner_'+str(num_rounds)+'.pdf', bbox_inches='tight')

# if args.dont_plot and args.ynorm:
#     print('option 1')
#     fig1 = corner(yscaler.inverse_transform(samples[-1]), smooth=0.5, range=prior_bounds)
#     plt.savefig(args.output+'corner_'+str(r+1)+'.pdf', bbox_inches='tight')
# elif args.dont_plot and not args.ynorm:
#     print('option 2')
#     fig1 = corner(samples[-1], smooth=0.5, range=prior_bounds)
#     plt.savefig(args.output+'corner_'+str(r+1)+'.pdf', bbox_inches='tight')

print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
