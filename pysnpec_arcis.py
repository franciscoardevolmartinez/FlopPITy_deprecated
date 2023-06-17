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
from sklearn.decomposition import PCA
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
    parser.add_argument('-output', type=str, default='output', help='Directory to save output')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: CPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    parser.add_argument('-samples_per_round', type=int, default=1000, help='Number of samples to draw for training each round. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=32)
    parser.add_argument('-transforms', type=int, default=5)
    parser.add_argument('-do_pca', action='store_true')
    parser.add_argument('-n_pca', type=int, default=50)
    parser.add_argument('-embed_size', type=int, default=64)
    parser.add_argument('-embedding', action='store_true')
    parser.add_argument('-bins', type=int, default=10)
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-ynorm', action='store_false')
    parser.add_argument('-xnorm', action='store_false')
    parser.add_argument('-Ztol', type=float, default=0.5)
    parser.add_argument('-convergence_criterion', type=str, default='absolute_error', help='Other options are: strictly_positive, ')
    parser.add_argument('-retrain_from_scratch', action='store_true')
    parser.add_argument('-Zrounds', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-nrepeat', type=int, default=3)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-atoms', type=int, default=100)
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-reuse_prior_samples', action='store_true')
    parser.add_argument('-prior_dir', type=str)
    parser.add_argument('-dont_reject', action='store_false')
    parser.add_argument('-npew', type=int, default=5000)
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
    if args.do_pca:
        if args.xnorm:
            default_x = xscaler.transform(pca.transform(obs.reshape(1,-1)))
        else:
            default_x = pca.transform(obs.reshape(1,-1))
    else:
        if args.xnorm:
            default_x = xscaler.transform(obs.reshape(1,-1))
        else:
            default_x = obs.reshape(1,-1)
    P = posterior.log_prob(torch.tensor(Y), x=default_x)
    pi = prior.log_prob(torch.tensor(Y))
    logZ =np.empty(3)
    logZ[0] = np.median(-(P-pi-L).detach().numpy())
    logZ[2] = np.percentile(-(P-pi-L).detach().numpy(), 84)
    logZ[1] = np.percentile(-(P-pi-L).detach().numpy(), 16)
    return logZ

### Embedding network
class SummaryNet(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()
        inter = int((size_in+size_out)/2)
        self.fc1 = nn.Linear(size_in, size_out)
        # self.fc2 = nn.Linear(inter, size_out)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        # x = self.fc2(x)
        return x
    
### Display 1D marginals in console
def post2txt(post, nbins=20, a=33, b=67):
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
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, nTpoints, nr,i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, nTpoints, nr,args.processes-1, len(obs), len(obs_spec)))
    else:
        for i in range(args.processes-1):
            parargs.append((params[i*samples_per_process:(i+1)*samples_per_process], args.output, r, args.input, freeT, 0, nr, i, len(obs), len(obs_spec)))
        parargs.append((params[(args.processes-1)*samples_per_process:], args.output, r, args.input, freeT, 0,nr, args.processes-1, len(obs), len(obs_spec)))

    tic=time()
    with Pool(processes = args.processes) as pool:
        arcis_specs = pool.starmap(simulator, parargs)
    arcis_spec = np.concatenate(arcis_specs)
    print('Time elapsed: ', time()-tic)
    logging.info(('Time elapsed: ', time()-tic))
    
    return arcis_spec

##### CREATE FOLDERS ######

# try:
p = Path(args.output)
p.mkdir(parents=True, exist_ok=True)
# except:
#     print('Folder already exists! Renaming to \''+args.output+'_new\'')
#     args.output+='_new'
#     p = Path(args.output)
#     p.mkdir(parents=True, exist_ok=False)

imgs = Path(args.output+'/Figures')
imgs.mkdir(parents=False, exist_ok=True)

arcis_logs = Path(args.output+'/ARCiS_logs')
arcis_logs.mkdir(parents=False, exist_ok=True)

##########################

logging.basicConfig(filename=args.output+'/log.log', filemode='a', format='%(asctime)s %(message)s', 
                datefmt='%H:%M:%S', level=logging.DEBUG)

logging.info('Initializing...')

logging.info('Command line arguments: '+ str(args))
print('Command line arguments: '+ str(args))

device = args.device

os.system('cp '+args.input + ' '+args.output+'/input_arcis.dat')

args.input = args.output+'/input_arcis.dat'

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

if args.embedding:
    print('Using an embedding network.')
    embedding_net = SummaryNet(obs_spec.shape[0], args.embed_size)
else:
    embedding_net = nn.Identity()
    
### Preprocessing for spectra and parameters
def preprocess(np_theta, arcis_spec):
    theta_aug = torch.tensor(np_theta, dtype=torch.float32, device=device)
    arcis_spec_aug = arcis_spec + noise_spec*np.random.randn(samples_per_round, obs_spec.shape[0])
    global xscaler
    global pca
    if r==0:
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
            print('No PCA, straight to xnorm')
            logging.info('No PCA, straight to xnorm')
            # print(arcis_spec)
            xscaler = StandardScaler().fit(arcis_spec)
            with open(args.output+'/xscaler.p', 'wb') as file_xscaler:
                pickle.dump(xscaler, file_xscaler)

    if args.do_pca:
        print('Doing PCA...')
        logging.info('Doing PCA...')
        x_i = pca.transform(arcis_spec_aug)
        if args.xnorm:
            x_f = xscaler.transform(x_i)
        else:
            x_f = x_i
        np.save(args.output+'/preprocessed.npy', x_f)
    elif args.xnorm:
        print('Normalizing features')
        logging.info('Normalizing features')
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

# neural_posterior = posterior_nn(model='nsf', hidden_features=args.hidden, num_transforms=args.transforms, num_bins=args.bins, num_blocks=args.blocks,
#                                 z_score_x='none', z_score_y='none', use_batch_norm=True, dropout_probability=args.dropout, embedding_net=embedding_net) #delete embedding_net #z_score='independent'

neural_posterior = posterior_nn(model='nsf', embedding_net=embedding_net, use_batch_norm=True, dropout_probability=args.dropout)

inference = SNPE_C(prior = prior, density_estimator=neural_posterior, device=device)  ### put this back when finished

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
    # 1st check where it was left off previously
    # check number of rounds run previously
    
    
    # Now read files
    print('Reading files from previous run...')
    logging.info('Reading files from previous run...')
    print('Reading Y.p ...')
    logging.info('Reading Y.p ...')
    np_theta = pickle.load(open(args.output+'/Y.p', 'rb'))
    print('Reading arcis_spec.p ...')
    logging.info('Reading arcis_spec.p ...')
    arcis_spec = pickle.load(open(args.output+'/arcis_spec.p', 'rb'))
    r=len(arcis_spec.keys())
    print('Reading posteriors.pt ...')
    logging.info('Reading posteriors.pt ...')
    posteriors=torch.load(args.output+'/posteriors.pt')
    proposal=posteriors[-1]
    print('Reading samples.p ...')
    logging.info('Reading samples.p ...')
    samples=pickle.load(open(args.output+'/samples.p', 'rb'))
    try:
        print('Reading evidence.p ...')
        logging.info('Reading evidence.p ...')
        logZs=pickle.load(open(args.output+'/evidence.p', 'rb'))
    except:
        print('oops')
    print('Reading inference.p ...')
    logging.info('Reading inference.p ...')
    inference=pickle.load(open(args.output+'/inference.p', 'rb'))
    inference_object=inference
    if args.xnorm:
        print('Reading xscaler.p ...')
        logging.info('Reading xscaler.p ...')
        xscaler=pickle.load(open(args.output+'/xscaler.p', 'rb'))
    print('Reading samples.p ...')
    logging.info('Reading samples.p ...')
    if args.ynorm:
        yscaler=pickle.load(open(args.output+'/yscaler.p', 'rb'))
    if args.do_pca:
        print('Reading pca.p ...')
        logging.info('Reading pca.p ...')
        pca=pickle.load(open(args.output+'/pca.p', 'rb'))
    num_rounds+=r
    
# samples_per_round_hack=[100000,5000]

good_rounds=[]

while r<num_rounds:
    print('\n')
    print('\n ##### Training round ', r)
    logging.info('#####  Round '+str(r)+'  #####')
    # samples_per_round = samples_per_round_hack[r]
    
    # samples_per_round=max(int(args.samples_per_round*(1/1+r)),100)
    
    if args.reuse_prior_samples and r==0:
        print('Reusing prior samples')
        logging.info('Reusing prior samples')
        try:
            arcis_spec[r] = pickle.load(open(args.prior_dir+'/arcis_spec.p', 'rb'))[0][:samples_per_round]
            np_theta[r] = pickle.load(open(args.prior_dir+'/Y.p', 'rb'))[0][:samples_per_round]
        except:
            arcis_spec[r] = np.load(args.prior_dir+'/arcis_spec_round_0.npy')
            np_theta[r] = np.load(args.prior_dir+'/Y_round_0.npy')
    else:
        
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
        with open(args.output+'/Figures/corner_'+str(r)+'.jpg', 'wb') as file_corner:
            plt.savefig(file_corner, bbox_inches='tight')
        plt.close('all')

        #### COMPUTE MODELS
        arcis_spec[r] = compute(np_theta[r])

        for j in range(args.processes):
            os.system('mv '+args.output + '/round_'+str(r)+str(j)+'_out/log.dat '+args.output +'/ARCiS_logs/log_'+str(r)+str(j)+'.dat')
            os.system('rm -rf '+args.output + '/round_'+str(r)+str(j)+'_out/')
            os.system('rm -rf '+args.output + '/round_'+str(r)+str(j)+'_samples.dat')

        #check if all models have been computed
        sm = np.sum(arcis_spec[r], axis=1)

        arcis_spec[r] = arcis_spec[r][sm!=0]
        np_theta[r] = np_theta[r][sm!=0]

        crash_count=0    
        while len(arcis_spec[r])<samples_per_round:
            crash_count+=1
            print('Crash ',str(crash_count))
            remain = samples_per_round-len(arcis_spec[r])
            print('ARCiS crashed, computing remaining ' +str(remain)+' models.')
            logging.info('ARCiS crashed, computing remaining ' +str(remain)+' models.')

            theta_ac = proposal.sample((remain,))
            np_theta_ac = theta_ac.cpu().detach().numpy().reshape([-1, len(prior_bounds)])

            arcis_spec_ac=compute(np_theta_ac)

            for j in range(args.processes):
                os.system('mv '+args.output + '/round_'+str(r)+str(j)+'_out/log.dat '+args.output +'/ARCiS_logs/log_'+str(r)+str(j)+str(crash_count)+'.dat')
                os.system('rm -rf '+args.output + '/round_'+str(r)+str(j)+'_out/')
                os.system('rm -rf '+args.output + '/round_'+str(r)+str(j)+'_samples.dat')

            sm_ac = np.sum(arcis_spec_ac, axis=1)

            arcis_spec[r] = np.concatenate((arcis_spec[r], arcis_spec_ac[sm_ac!=0]))
            np_theta[r] = np.concatenate((np_theta[r], np_theta_ac[sm_ac!=0]))

    ### COMPUTE EVIDENCE
    if r>0:
        logging.info('Computing evidence...')
        logZ = evidence(posteriors[-1], prior, arcis_spec[r], np_theta[r], obs_spec, noise_spec)
        print('\n')
        print('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        logging.info('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        print('\n')
        logZs.append(logZ)

    if args.dont_reject and r>1 and logZs[-1][0]<logZs[-2][0] and logZs[-1][1]<logZs[-2][1]: #change logZs[-2][2] to logZs[-2][0]
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
        print('Preprocessing data...')
        logging.info('Preprocessing data...')
        theta_aug, x = preprocess(np_theta[r], arcis_spec[r])
        reject=False
        if r>1 and abs(logZs[-1][0]-logZs[-2][0])<args.Ztol:
            print('ΔZ < Ztol')
            logging.info('ΔZ < Ztol')
            good_rounds.append(1)
            if sum(good_rounds[-args.Zrounds:])==args.Zrounds:
                print('Last '+str(args.Zrounds)+'rounds have had ΔZ < Ztol. Ending inference.')
                logging.info('Last '+str(args.Zrounds)+'rounds have had ΔZ < Ztol. Ending inference.')
                r=num_rounds
            elif sum(good_rounds[-args.Zrounds:])<args.Zrounds:
                r+=1
        else:
            r+=1
            good_rounds.append(0)

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


    ### IF ROUND IS REJECTED, 'IMPROVEMENT' IN TRAINING SHOULD ALSO BE REJECTED ####
    ###  
    ###   FIX THIS!!!!!!

    with open(args.output+'/x.p', 'wb') as file_x:
        pickle.dump(x, file_x)

    if not reject:
        inference_object = inference.append_simulations(theta_aug, x, proposal=proposal)
        with open(args.output+'/inference.p', 'wb') as file_inference:
            pickle.dump(inference, file_inference)

    ##################################################

    posterior_estimator = inference_object.train(show_train_summary=True, stop_after_epochs=args.patience, num_atoms=args.atoms, force_first_round_loss=True, retrain_from_scratch=args.retrain_from_scratch, use_combined_loss=True) #use_combined_loss

    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.plot()
    with open(args.output+'/Figures/corner_'+str(r)+'.jpg', 'wb') as file_corner:
        plt.savefig(file_corner, bbox_inches='tight')
    plt.close('all')

    ### GENERATE POSTERIOR

    if args.do_pca:
        print('Transforming observation...')
        logging.info('Transforming observation...')
        default_x_pca = pca.transform(obs_spec.reshape(1,-1))
        if args.xnorm:
            default_x = xscaler.transform(default_x_pca)
        else:
            default_x = default_x_pca
    elif args.xnorm:
        default_x = xscaler.transform(obs_spec.reshape(1,-1))
    else:
        default_x = obs_spec.reshape(1,-1)

    print('\n Time elapsed: '+str(time()-tic))
    logging.info('Time elapsed: '+str(time()-tic))
    posterior = inference_object.build_posterior(sample_with='rejection').set_default_x(default_x)
    posteriors.append(posterior)
    print('Saving posteriors ')
    logging.info('Saving posteriors ')
    with open(args.output+'/posteriors.pt', 'wb') as file_posteriors:
        torch.save(posteriors, file_posteriors)
    proposal = posterior


    ### DRAW 5000 SAMPLES (JUST FOR PLOTTING)

    if not reject:
        print('Saving samples ')
        logging.info('Saving samples ')
        tsamples = posterior.sample((args.npew,))
        if args.ynorm:
            samples.append(yscaler.inverse_transform(tsamples.cpu().detach().numpy()))
        else:
            samples.append(tsamples.cpu().detach().numpy())

        with open(args.output+'/samples.p', 'wb') as file_samples:
            pickle.dump(samples, file_samples)

    print('\n')
    print('##### 1D marginals #####')
    print('________________________')
    print('\n')
    post2txt(samples[-1])
    print('\n')

    plt.close('all')
    

### FINISHING    
    
print('Inference ended.')    
logging.info('Inference ended.')

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

fig1 = corner(samples[-1], color='rebeccapurple', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
with open(args.output+'/Figures/corner_'+str(r)+'.jpg', 'wb') as file_post_equal_corner:
    plt.savefig(file_post_equal_corner, bbox_inches='tight')
plt.close('all')

print('\n')
print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
