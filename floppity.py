import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchmetrics.functional import kl_divergence
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import numpy as np
from sbi import utils as utils
from sbi.inference import SNPE_C
from time import time
import logging
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from corner import corner
from pysnpec_arcis import *

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
    parser.add_argument('-twoterms', action='store_true')
    parser.add_argument('-blocks', type=int, default=2)
    parser.add_argument('-ynorm', action='store_true')
    parser.add_argument('-xnorm', action='store_true')
    parser.add_argument('-Ztol', type=float, default=0.5)
    parser.add_argument('-convergence_criterion', type=str, default='absolute_error', help='Other options are: strictly_positive, ')
    parser.add_argument('-retrain_from_scratch', action='store_true')
    parser.add_argument('-Zrounds', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-nrepeat', type=int, default=3)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-atoms', type=int, default=100)
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-reuse_prior_samples', action='store_true')
    parser.add_argument('-prior_dir', type=str)
    parser.add_argument('-dont_reject', action='store_false')
    parser.add_argument('-npew', type=int, default=5000)
    return parser.parse_args()

args = parse_args()

work_dir = os.getcwd()+'/' #Get directory


##### CREATE FOLDERS TO SAVE OUTPUT
p = Path(args.output)
p.mkdir(parents=True, exist_ok=True)

imgs = Path(args.output+'/Figures')
imgs.mkdir(parents=False, exist_ok=True)

arcis_logs = Path(args.output+'/ARCiS_logs')
arcis_logs.mkdir(parents=False, exist_ok=True)
###################################


##### INITIALIZE LOGGING
logging.basicConfig(filename=args.output+'/log.log', filemode='a', format='%(asctime)s %(message)s', 
                datefmt='%H:%M:%S', level=logging.DEBUG)

logging.info('Initializing...')
print('Initializing...')

logging.info('Command line arguments: '+ str(args))
print('Command line arguments: '+ str(args))
#######################


##### READ ARCiS INPUT FILE
os.system('cp '+args.input + ' '+args.output+'/input_arcis.dat')

args.input = args.output+'/input_arcis.dat'

parnames, prior_bounds, obs, obs_spec, noise_spec, nr = read_arcis_input(args.input, args.twoterms)
#####################


##### EMBEDDING NETWORK (useless atm)
if args.embedding:
    print('Using an embedding network.')
    summary = SummaryNet(obs_spec.shape[0], args.embed_size)
else:
    summary = nn.Identity()
#######################


##### PRIOR 
if args.ynorm:
    yscaler = Normalizer(prior_bounds)
    with open(args.output+'/yscaler.p', 'wb') as file_yscaler:
        pickle.dump(yscaler, file_yscaler)

    prior_min = torch.tensor(yscaler.transform(prior_bounds[:,0].reshape(1, -1)).reshape(-1))
    prior_max = torch.tensor(yscaler.transform(prior_bounds[:,1].reshape(1, -1)).reshape(-1))
else:
    prior_min = torch.tensor(prior_bounds[:,0].reshape(1,-1))
    prior_max = torch.tensor(prior_bounds[:,1].reshape(1,-1))

prior = utils.BoxUniform(low=prior_min.to(args.device, non_blocking=True), high=prior_max.to(args.device, non_blocking=True), device=args.device)
#######################


num_rounds = args.num_rounds
samples_per_round = args.samples_per_round

inference = SNPE_C(prior = prior, density_estimator='nsf', device=args.device)

proposal=prior
posteriors=[]

samples=[]

logZs = []

good_rounds=[]

np_theta = {}
arcis_spec = {}

r=0
repeat=0

model_time=[]
train_time=[]


##### LOAD FILES IF RESUMING
if args.resume:
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
    
while r<num_rounds:
    print('\n')
    print('\n ##### Training round ', r)
    logging.info('#####  Round '+str(r)+'  #####')
    
    if args.reuse_prior_samples and r==0:
        print('Reusing prior samples')
        logging.info('Reusing prior samples')
        arcis_spec[r] = pickle.load(open(args.prior_dir+'/arcis_spec.p', 'rb'))[0][:samples_per_round]
        np_theta[r] = pickle.load(open(args.prior_dir+'/Y.p', 'rb'))[0][:samples_per_round]
    else:
        ##### DRAW SAMPLES
        logging.info('Drawing '+str(samples_per_round)+' samples')
        print('Samples per round: ', samples_per_round)
        theta = proposal.sample((samples_per_round,))
        np_theta[r] = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
        
        
        ##### NORMALIZE IF NECESSARY
        if args.ynorm:
            post_plot = yscaler.inverse_transform(np_theta[r])
        else:
            post_plot = np_theta[r]
        
        
        ##### CREATE CORNER PLOT
        fig1 = corner(post_plot, color='skyblue', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
        with open(args.output+'/Figures/corner_'+str(r)+'.jpg', 'wb') as file_corner:
            plt.savefig(file_corner, bbox_inches='tight')
        plt.close('all')

        
        ##### COMPUTE MODELS
        tic_compute=time()
        if args.twoterms:
            arcis_spec[r] = compute_2term(np_theta[r])
        else:
            arcis_spec[r] = compute(np_theta[r], args.processes, args.output,args.input, args.ynorm, r, nr, obs, obs_spec)
        
          
        ##### CHECK IF ALL MODELS WERE COMPUTED AND COMPUTE REMAINING IF NOT
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

            arcis_spec_ac=compute(np_theta_ac, args.processes, args.output,args.input, args.ynorm, r, nr, obs, obs_spec)

            sm_ac = np.sum(arcis_spec_ac, axis=1)

            arcis_spec[r] = np.concatenate((arcis_spec[r], arcis_spec_ac[sm_ac!=0]))
            np_theta[r] = np.concatenate((np_theta[r], np_theta_ac[sm_ac!=0]))
            
        model_time.append(time()-tic_compute)  
        logging.info('Time elapsed: '+ str(time()-tic_compute))
        print('Time elapsed: ', time()-tic_compute)

            
    ##### COMPUTE EVIDENCE
    if r>0:
        logging.info('Computing evidence...')
        logZ = evidence(posteriors[-1], prior, arcis_spec[r], np_theta[r], obs_spec, noise_spec, args.do_pca, args.xnorm, xscaler, pca)
        print('\n')
        print('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        logging.info('ln (Z) = '+ str(round(logZ[0], 2))+' ('+str(round(logZ[1],2))+', '+str(round(logZ[2],2))+')')
        print('\n')
        logZs.append(logZ)
    
    
    ###### REJECT ROUND IF NECESSARY
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
        print('Preprocessing data...')
        logging.info('Preprocessing data...')
        theta_aug, x, xscaler, pca = preprocess(np_theta[r], arcis_spec[r], r, samples_per_round, obs_spec, noise_spec, args.do_pca, args.n_pca, args.xnorm, args.output, args.device)
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

            
    ###### SAVE TRAINING EXAMPLES
    logging.info('Saving training examples...')
    print('Saving training examples...')
    with open(args.output+'/arcis_spec.p', 'wb') as file_arcis_spec:
        pickle.dump(arcis_spec, file_arcis_spec)
    with open(args.output+'/Y.p', 'wb') as file_np_theta:
        pickle.dump(np_theta, file_np_theta)
        
        
    ##### TRAIN
    tic = time()
    logging.info('Training SNPE...')
    print('Training SNPE...')

    ### IF ROUND IS REJECTED, 'IMPROVEMENT' IN TRAINING SHOULD ALSO BE REJECTED ####
    ###  
    ###   FIX THIS!!!!!!
    
    # Only append training examples if the round is not rejected
    if not reject:
        inference_object = inference.append_simulations(theta_aug, x, proposal=proposal)
        with open(args.output+'/inference.p', 'wb') as file_inference:
            pickle.dump(inference, file_inference)

    posterior_estimator = inference_object.train(show_train_summary=True, stop_after_epochs=args.patience, num_atoms=args.atoms, force_first_round_loss=True, retrain_from_scratch=args.retrain_from_scratch, use_combined_loss=True) #use_combined_loss


    ##### GENERATE POSTERIOR AND UPDATE PROPOSAL
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
    train_time.append(time()-tic)
    posterior = inference_object.build_posterior(sample_with='rejection').set_default_x(default_x)
    posteriors.append(posterior)
    print('Saving posteriors ')
    logging.info('Saving posteriors ')
    with open(args.output+'/posteriors.pt', 'wb') as file_posteriors:
        torch.save(posteriors, file_posteriors)
    proposal = posterior
    
    
    ##### CALCULATE KL DIVERGENCE (SEE HOW MUCH THE POSTERIOR CHANGED FROM LAST ROUND)
    if len(posteriors)>1:
        p = posteriors[-1].log_prob(torch.tensor(np_theta[r-1]), x=default_x).reshape(1,-1)
        q = posteriors[-2].log_prob(torch.tensor(np_theta[r-1]), x=default_x).reshape(1,-1)
        KL = kl_divergence(p,q, log_prob=True)

        print('KL divergence:', KL)


    ### DRAW npew SAMPLES (JUST FOR PLOTTING)
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
    post2txt(samples[-1], parnames,prior_bounds)
    print('\n')
    
    
print('Inference ended.')    
logging.info('Inference ended.')

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

fig1 = corner(samples[-1], color='skyblue', show_titles=True, smooth=0.9, range=prior_bounds, labels=parnames)
with open(args.output+'/Figures/corner_'+str(r)+'.jpg', 'wb') as file_post_equal_corner:
    plt.savefig(file_post_equal_corner, bbox_inches='tight')
plt.close('all')

print('\n')
print('Time elapsed: ', time()-supertic)
logging.info('Time elapsed: '+str(time()-supertic))
print('\n')
print('Time elapsed computing models: ', sum(model_time))
logging.info('Time elapsed computing models: '+ str(sum(model_time)))
print('\n')
print('Time elapsed training: ', sum(train_time))
logging.info('Time elapsed training: '+ str(sum(train_time)))