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

supertic = time()

### PARSE COMMAND LINE ARGUMENTS ###

def parse_args():
    parser = argparse.ArgumentParser(description=('Train SNPE_C'))
    parser.add_argument('-input', type=str, help='ARCiS input file for the retrieval')
    parser.add_argument('-obs', type=str, default='obs1.txt', help='File with observation')
    parser.add_argument('-output', type=str, default='output/', help='Directory to save output')
    parser.add_argument('-prior', type=str, default='prior.dat', help='File with prior bounds (box uniform)')
    parser.add_argument('-model', type=str, default='nsf', help='Either nsf or maf.')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: GPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    parser.add_argument('-samples_per_round', type=str, default='1000', help='Number of samples to draw for training each round. If a single value, that number of samples will be drawn for num_rounds rounds. If multiple values, at each round, the number of samples specified will be drawn. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=50)
    parser.add_argument('-do_pca', action='store_true')
    parser.add_argument('-n_pca', type=int, default=50)
    parser.add_argument('-transforms', type=int, default=5)
    parser.add_argument('-bins', type=int, default=5)
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-embed_size', type=int, default=32)
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
    return parser.parse_args()

### CREATE ARCIS SIMULATOR ###

class SummaryNet(nn.Module): 

    def __init__(self, size_in, size_out): 
        super().__init__()
        self.fc1 = nn.Linear(size_in, size_out)
#        self.fc2 = nn.Linear(256, 128)
#        self.fc3 = nn.Linear(128, 64)

    def forward(self, x):
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x
    
work_dir = os.getcwd()+'/'

args = parse_args()

p = Path(args.output)
p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=args.output+'/log.log', filemode='a', format='%(asctime)s %(message)s', 
                datefmt='%H:%M:%S', level=logging.DEBUG)

logging.info('Initializing...')

if args.embedding:
        print('Embedding net')

logging.info('Command line arguments: '+ str(args))
print('Command line arguments: '+ str(args))

device = args.device

# xscaler = pickle.load(open('xscaler.p', 'rb'))
# yscaler = pickle.load(open('yscaler.p', 'rb'))
# noise   = np.loadtxt('noise.dat')

### Normalize parameters

def scaleR(obs, model):
    return np.sum(obs)/np.sum(model, axis=1)

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
    
def removeR(X):
    means = np.mean(X, axis=0)
    XmR = X - means
    return np.concatenate((XmR, means), axis=1)
    
    
print('Loading prior from '+args.prior)
logging.info('Loading prior from '+args.prior)
    
prior_bounds = np.loadtxt(args.prior)

if args.ynorm:
    yscaler = Normalizer(prior_bounds)
    # yscaler = StandardScaler().fit()
    pickle.dump(yscaler, open(args.output+'/yscaler.p', 'wb'))

    prior_min = torch.tensor(yscaler.transform(prior_bounds[:,0].reshape(1, -1)).reshape(-1))
    prior_max = torch.tensor(yscaler.transform(prior_bounds[:,1].reshape(1, -1)).reshape(-1))
else:
    prior_min = torch.tensor(prior_bounds[:,0].reshape(1,-1))
    prior_max = torch.tensor(prior_bounds[:,1].reshape(1,-1))

prior = utils.BoxUniform(low=prior_min.to(device, non_blocking=True), high=prior_max.to(device, non_blocking=True), device=device)

## Define simulator and load observation (for noise)

x_o = np.loadtxt(args.obs)

embedding_net = SummaryNet(x_o.shape[0]+1 if args.removeR else x_o.shape[0], args.embed_size)

num_rounds = args.num_rounds

if args.resume:
    posteriors = []                         
    proposal = prior
else:
    posteriors = torch.load(args.output+'/posteriors.pt')
    if args.xnorm and not args.do_pca:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        proposal = posteriors[-1].set_default_x(xscaler.transform(x_o[:,1].reshape(1,-1)))
    elif args.do_pca and not args.xnorm:
        pca = pickle.load(open(args.output+'/pca.p', 'rb'))
        proposal = posteriors[-1].set_default_x(pca.transform(x_o[:,1].reshape(1,-1)))
    elif args.do_pca and args.xnorm:
        xscaler = pickle.load(open(args.output+'/xscaler.p', 'rb'))
        pca = pickle.load(open(args.output+'/pca.p', 'rb'))
        proposal = posteriors[-1].set_default_x(xscaler.transform(pca.transform(x_o[:,1].reshape(1,-1))))
    else:
        proposal = posteriors[-1].set_default_x(x_o[:,1])

# simulator, prior = prepare_for_sbi(arcis_sim, prior)

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
    logging.info('Drawing '+str(samples_per_round[r])+' samples')
    theta = proposal.sample((samples_per_round[r],))
    # print('itheta', itheta[0:2])
    np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
    
    if args.dont_plot and args.ynorm:
        fig1 = corner(yscaler.inverse_transform(np_theta), smooth=0.5, range=prior_bounds)
        plt.savefig(args.output+'corner_'+str(r)+'.pdf', bbox_inches='tight')
    elif args.dont_plot:
        fig1 = corner(np_theta, smooth=0.5, range=prior_bounds)
        plt.savefig(args.output+'corner_'+str(r)+'.pdf', bbox_inches='tight')
    
    samples_per_process = samples_per_round[r]//args.processes
    
    parargs=[]
    if args.ynorm:
        params=yscaler.inverse_transform(np_theta)
    else:
        params = np_theta
        
    for i in range(args.processes-1):
        parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, args.obs, i))
    parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, args.obs, args.processes-1))
    
    tic=time()
    pool = Pool(processes = args.processes)
    Xes = pool.starmap(simulator, parargs)
    print('Time elapsed: ', time()-tic)
    logging.info(('Time elapsed: ', time()-tic))
    
#     ### CALL TO ARCIS
#     fname = args.output+'round_'+str(r)+'_samples.dat'
#     # y_nat[:,2:5] = 10**y_nat[:,2:5]
#     if args.ynorm:
#         np.savetxt(fname, yscaler.inverse_transform(np_theta))
#         if args.dont_plot:
#             fig1 = corner(yscaler.inverse_transform(np_theta), smooth=0.5, range=prior_bounds)
#             plt.savefig(args.output+'corner_'+str(r)+'.pdf', bbox_inches='tight')
#     else:
#         np.savetxt(fname, np_theta)
#         if args.dont_plot:
#             fig1 = corner(np_theta, smooth=0.5, range=prior_bounds)
#             plt.savefig(args.output+'corner_'+str(r)+'.pdf', bbox_inches='tight')
        
#     print('Running ARCiS')
#     logging.info('Running ARCiS')
#     os.system('cd .. ; '+ARCiS + ' '+args.input + ' -o '+args.output+'round_'+str(r)+'_out -s parametergridfile='+fname+' -s obs01:file='+args.obs)
    
#     X = np.zeros([samples_per_round[r],x_o.shape[0]])

#     dirx = args.output + 'round_'+str(r)+'_out/'
    
#     print('Reading ARCiS output')
#     logging.info('Reading ARCiS output')
    
#     for i in trange(samples_per_round[r]):
#         if i+1<10:
#             model_dir = dirx + 'model00000'+str(i+1)
#         elif i+1<100:
#             model_dir = dirx + 'model0000'+str(i+1)
#         elif i+1<1000:
#             model_dir = dirx + 'model000'+str(i+1)
#         elif i+1<1e4:
#             model_dir = dirx + 'model00'+str(i+1)
#     #     print(model_dir)
#         try:
#             X[i] = np.loadtxt(model_dir+'/trans')[:,1]# + x_o[:,2]*np.random.randn(1, x_o.shape[0])
#         except:
#             print(model_dir)

    X = np.concatenate(Xes)
    
    if args.scaleR:
        scale_R = R_scale(x_o[:,1], X)
        
        scaled_R = np.sqrt(scale_R)*np_theta[:,-2]
        np_theta[:,-2] = scaled_R
    
    np.save(args.output+'/X_round_'+str(r)+'.npy', X)
            
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
    
    # theta, x = simulate_for_sbi(simulator, proposal, num_simulations=samples_per_round[r], show_progress_bar=False) 
    
    theta = torch.tensor(np.repeat(np_theta, args.naug, axis=0), dtype=torch.float32, device=device)
    X_aug = np.repeat(X, args.naug, axis=0) + x_o[:,2]*np.random.randn(samples_per_round[r]*args.naug, x_o.shape[0])
    if r==0:
        if args.xnorm and not args.do_pca:
            xscaler = StandardScaler().fit(X)
            pickle.dump(xscaler, open(args.output+'/xscaler.p', 'wb'))
        elif args.do_pca and not args.xnorm:
            pca = PCA(n_components=args.n_pca)
            pca.fit(X)
            pickle.dump(pca, open(args.output+'/pca.p', 'wb'))
        elif args.do_pca and args.xnorm:
            #do pca first
            pca = PCA(n_components=args.n_pca)
            pca.fit(X)
            pickle.dump(pca, open(args.output+'/pca.p', 'wb'))
            #then do xnorm
            xscaler = StandardScaler().fit(pca.transform(X))
            pickle.dump(xscaler, open(args.output+'/xscaler.p', 'wb'))
    ##adding comments
    if args.xnorm and not args.do_pca:
        x = torch.tensor(xscaler.transform(X_aug), dtype=torch.float32, device=device)
    elif args.do_pca and not args.xnorm:
        x = torch.tensor(pca.transform(X_aug), dtype=torch.float32, device=device)
    elif args.do_pca and args.xnorm:
        x = torch.tensor(xscaler.transform(pca.transform(X_aug)), dtype=torch.float32, device=device)
    else:
        x = torch.tensor(X_aug, dtype=torch.float32, device=device)
    
    # ### This is just for debugging
    # if args.dont_plot:
    #     plt.figure(figsize=(7,5))
    #     for i in range(100):
    #         plt.plot(x_o[:,0], x.detach().numpy()[i], color='midnightblue', alpha=0.1)
    #     plt.savefig(args.output+'x_'+str(r)+'.pdf', bbox_inches='tight')
    
    logging.info('Training...')
    tic = time()
    if args.method=='snpe':
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(
            discard_prior_samples=args.discard_prior_samples, use_combined_loss=args.combined, show_train_summary=True, 
            stop_after_epochs=args.patience, num_atoms=args.atoms, retrain_from_scratch=args.retrain_from_scratch)
    elif args.method=='snle':
        density_estimator = inference.append_simulations(theta, x).train(
            discard_prior_samples=args.discard_prior_samples, show_train_summary=True, 
            stop_after_epochs=args.patience)

    print('\n Time elapsed: '+str(time()-tic))
    logging.info('Time elapsed: '+str(time()-tic))
    try:
        posterior = inference.build_posterior(density_estimator)
    except:
        print('\n OH NO!, IT HAPPENED!')
        logging.info('OH NO!, IT HAPPENED!')
        try:
            posterior = inference.build_posterior(density_estimator)
        except:
            print('\n OH NO!, IT HAPPENED *AGAIN*!?')
            logging.info('OH NO!, IT HAPPENED *AGAIN*!?')
            posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    print('Saving posteriors ')
    logging.info('Saving posteriors ')
    torch.save(posteriors, args.output+'/posteriors.pt')
    
    if args.xnorm and not args.do_pca:
        proposal = posterior.set_default_x(xscaler.transform(x_o[:,1].reshape(1,-1)))
    elif args.do_pca and not args.xnorm:
        proposal = posterior.set_default_x(pca.transform(x_o[:,1].reshape(1,-1)))
    elif args.xnorm and args.do_pca:
        proposal = posterior.set_default_x(xscaler.transform(pca.transform(x_o[:,1].reshape(1,-1))))
    else:
        proposal = posterior.set_default_x(x_o[:,1])
    if args.clean:
        for j in range(args.processes):
            os.system('rm -rf '+args.output + 'round_'+str(r)+str(j)+'_out/')

print('Drawing samples ')
logging.info('Drawing samples ')
samples = []
for j in range(num_rounds):
    print('Drawing samples from round ', j)
    logging.info('Drawing samples from round ' + str(j))
    if args.xnorm and not args.do_pca:
        obs = torch.tensor(xscaler.transform(x_o[:,1].reshape(1,-1)), device=device)
    elif args.do_pca and not args.xnorm:
        obs = torch.tensor(pca.transform(x_o[:,1].reshape(1,-1)), device=device)
    elif args.do_pca and args.xnorm:
        obs = torch.tensor(xscaler.transform(pca.transform(x_o[:,1].reshape(1,-1))), device=device)
    else:
        obs = torch.tensor(x_o[:,1].reshape(1,-1), device=device)
    posterior = posteriors[j]
    tsamples = posterior.sample((2000,), x=obs, show_progress_bars=True)
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
