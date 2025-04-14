import argparse
from pathlib import Path
import torch
# from torchmetrics.functional import kl_divergence
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import numpy as np
from sbi import utils as utils
import logging
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from floppityFUN import *
from simulator import *
from modules import *

# os.system('cd MulteXBI; git rev-parse HEAD > MulteXBI/gitversion.txt')

version = np.loadtxt('MulteXBI/gitversion.txt', dtype='str')

### PARSE COMMAND LINE ARGUMENTS ###
def parse_args():
    parser = argparse.ArgumentParser(description=('Train SNPE_C'))
    parser.add_argument('-input', type=str, help='ARCiS input file for the retrieval')
    parser.add_argument('-input2', type=str, default='aintnothinhere', help='ARCiS input file for a 2nd limb')
    parser.add_argument('-fit_frac', action='store_true', help='If true, fit for the relative contribution of the 2 components (input and input2).')
    parser.add_argument('-n_global', type=int, default=5, help='Number of global parameters. Only used in conjunction with input2.')
    parser.add_argument('-binary', action='store_true', help='If true, the two components (input and input2) are added together instead of averaged.')
    parser.add_argument('-output', type=str, default='output', help='Directory to save output')
    parser.add_argument('-device', type=str, default='cpu', help='Device to use for training. Default: CPU.')
    parser.add_argument('-num_rounds', type=int, default=10, help='Number of rounds to train for. Default: 10.')
    # parser.add_argument('-forward', type=str, default='ARCiS')
    parser.add_argument('-samples_per_round', type=int, default=1000, help='Number of samples to draw for training each round. Default: 1000.')
    parser.add_argument('-hidden', type=int, default=50, help='Number of neurons per layer in the ResNet')
    parser.add_argument('-transforms', type=int, default=10)
    parser.add_argument('-tail_bound', type=float, default=3.0)
    # parser.add_argument('-custom_nsf', action='store_true')
    parser.add_argument('-do_pca', action='store_true')
    parser.add_argument('-rem_mean', action='store_true')
    parser.add_argument('-res_scale', action='store_true')
    parser.add_argument('-naug', type=int, default=1, help='Data augmentation factor')
    parser.add_argument('-n_pca', type=int, default=50)
    parser.add_argument('-flow', type=str, default='nsf')
    parser.add_argument('-embed_size', type=str, default='64')
    parser.add_argument('-embedding', action='store_true')
    parser.add_argument('-embedding_type', type=str, default='CNN', help='Can be FC, CNN or multi')
    parser.add_argument('-embed_hypers', type=str, default='2, 4, 2, 64, 5')
    parser.add_argument('-bins', type=int, default=8)
    parser.add_argument('-twoterms', action='store_true')
    parser.add_argument('-blocks', type=int, default=3)
    parser.add_argument('-ynorm', action='store_true')
    parser.add_argument('-xnorm', action='store_true')
    # parser.add_argument('-Ztol', type=float, default=0.5)
    # parser.add_argument('-convergence_criterion', type=str, default='absolute_error', help='Other options are: strictly_positive, ')
    parser.add_argument('-retrain_from_scratch', action='store_true')
    # parser.add_argument('-Zrounds', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.1)
    # parser.add_argument('-max_reject', type=int, default=3)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-patience', type=int, default=10)
    parser.add_argument('-atoms', type=int, default=10)
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-reuse_prior_samples', action='store_true')
    parser.add_argument('-prior_dir', type=str)
    # parser.add_argument('-dont_reject', action='store_false')
    parser.add_argument('-npew', type=int, default=5000)
    parser.add_argument('-fit_offset', action='store_true')
    parser.add_argument('-max_offset', type=float, default=1e-2)
    parser.add_argument('-fit_scaling', action='store_true')
    parser.add_argument('-max_scaling', type=float, default=0.1)
    parser.add_argument('-fit_vrot', action='store_true')
    parser.add_argument('-min_vrot', type=float)
    parser.add_argument('-max_vrot', type=float)
    return parser.parse_args()

args = parse_args()

work_dir = os.getcwd()+'/' #Get directory

##### CREATE FOLDERS TO SAVE OUTPUT
p = Path(args.output)
p.mkdir(parents=True, exist_ok=True)

imgs = Path(args.output+'/Figures')
imgs.mkdir(parents=False, exist_ok=True)

logs = Path(args.output+'/ARCiS_logs')
logs.mkdir(parents=False, exist_ok=True)


##### INITIALIZE LOGGING
logging.basicConfig(filename=args.output+'/log.log', filemode='a', format='%(asctime)s %(message)s', 
                datefmt='%H:%M:%S', level=logging.DEBUG)


add_log(f'Running FlopPITy (c)Francisco Ardévol Martínez\n')
add_log(f'version {version}\n')
add_log('Command line arguments: '+ str(args))


##### READ ARCiS INPUT FILE
parnames, prior_bounds, obs, wvl_spec, obs_spec, noise_spec, nr, which, init2, nwvl, log, arcis_par = read_input(args)


#######  REMOVE MEAN? Useful for transmission spectra
if args.rem_mean:
    rem_mean=rm_mean()
else:
    rem_mean=do_nothing()


##### EMBEDDING NETWORK
summary = createEmbedding(args.embedding, args.embedding_type, args.embed_size, args.embed_hypers, args.rem_mean, obs_spec.shape[0])


##### PRIOR 
if args.ynorm:
    yscaler = Normalizer(prior_bounds, args.tail_bound)
    with open(args.output+'/yscaler.p', 'wb') as file_yscaler:
        pickle.dump(yscaler, file_yscaler)

prior = create_prior(prior_bounds, args.ynorm, yscaler, args.tail_bound, args.device, args.output)


####### CREATE q(θ|x) 

inference = density_builder(args.flow, args.transforms, args.hidden, args.bins, summary, args.blocks, args.dropout, prior, args.device)

proposal=prior


##### INITIALIZING LISTS #####
posteriors=[]

samples=[]

logZs = []

w_is = []

neffs = []

good_rounds=[]

np_theta = {}
arcis_spec = {}
T = {}

r=0
repeat=0

model_time=[]
train_time=[]

num_rounds = args.num_rounds
samples_per_round = args.samples_per_round


##### LOAD FILES IF RESUMING
if args.resume:
    np_theta, arcis_spec, posteriors, proposal, samples, logZs, inference, inference_object, xscaler, yscaler, pca, r, num_rounds = load_files(args.output, args.xnorm, args.ynorm, args.do_pca, args.num_rounds)
    
while r<num_rounds:
    add_log('#####  Round '+str(r)+'  #####')
    
    if args.reuse_prior_samples and r==0:
        add_log('Reusing prior samples')
        arcis_spec[r] = pickle.load(open(args.prior_dir+'/arcis_spec.p', 'rb'))[0][:samples_per_round]
        np_theta[r] = pickle.load(open(args.prior_dir+'/Y.p', 'rb'))[0][:samples_per_round]
    else:
        
        ##### GENERATE TRAINING EXAMPLES:
        ##### 1. Draw samples (θ)
        np_theta[r] = sample_proposal(r, proposal, samples_per_round, prior_bounds, args.tail_bound, args.input2, 
                                      args.n_global, init2)
        
        ##### Compute models (x)
        arcis_spec[r] = compute(np_theta[r], args.processes, args.output, args.input, args.input2, args.n_global, which, 
                                args.ynorm, yscaler, r, nr, obs, wvl_spec, obs_spec,nwvl,arcis_par, parnames, args)
        
        #### Check that all models were computed
        arcis_spec[r], np_theta[r] = check_crash(arcis_spec[r], np_theta[r], samples_per_round, proposal, prior_bounds,  yscaler, which, r, nr, obs, wvl_spec, obs_spec, nwvl, arcis_par, parnames,args)
        
          
    ##### PREPROCESS DATA
    theta_aug, x, xscaler, pca = preprocess(np_theta[r], arcis_spec[r], r, samples_per_round, obs_spec, noise_spec, args.naug,
                                            args.do_pca, args.n_pca, args.xnorm, nwvl, args.rem_mean, args.output, args.device, args)
    
    ##### Save models and parameters
    add_log('Saving training examples...')
    with open(args.output+'/arcis_spec.p', 'wb') as file_arcis_spec:
        pickle.dump(arcis_spec, file_arcis_spec)
    with open(args.output+'/Y.p', 'wb') as file_np_theta:
        pickle.dump(np_theta, file_np_theta)
        
    ### COMPUTE IS efficiency
    if r>0:
        w_i, eff = IS(posteriors[-1], obs_spec, noise_spec, arcis_spec[r], np_theta[r])
        logZ_is, logZ_err_is = evidence_from_IS(w_i, eff)

        add_log('IS efficiency: ' + str(np.round(eff, 3)))
        add_log('ln (Z) = '+ str(round(logZ_is, 2))+' +- '+str(round(logZ_err_is,2)))
    
        w_is.append(w_i)
        neffs.append(eff)
    
    ##### TRAIN NSF
    # inference_object = inference.append_simulations(theta_aug, x, proposal=proposal)
    inference.append_simulations(theta_aug, x, proposal=proposal)

    # current_inference_object = inference_object
    posterior_estimator = inference.train(show_train_summary=True, stop_after_epochs=args.patience,
                                                         num_atoms=args.atoms,force_first_round_loss=True,
                                                         retrain_from_scratch=args.retrain_from_scratch, use_combined_loss=True) 

    default_x = xscaler.transform(pca.transform(rem_mean.transform(obs_spec.reshape(1,-1))))

    posterior = inference.build_posterior(posterior_estimator).set_default_x(default_x)
        
    
    posteriors.append(posterior)
    # inference_object=current_inference_object
    proposal = posterior

    
    ##### Save files
    if r>0:
        with open(args.output+'/importance_weights.p', 'wb') as file_w_is:
            add_log('Saving importance weights...')
            pickle.dump(w_is, file_w_is)
        with open(args.output+'/importance_effs.p', 'wb') as file_neffs:
            add_log('Saving importance efficiencies...')
            pickle.dump(neffs, file_neffs)

    add_log('Saving posteriors ')
    with open(args.output+'/posteriors.pt', 'wb') as file_posteriors:
        torch.save(posteriors, file_posteriors)

    with open(args.output+'/inference.p', 'wb') as file_inference:
        pickle.dump(inference, file_inference)
 
    ##### Save samples
    add_log('Saving samples ')
    tsamples = proposal.sample((args.npew,))

    if args.ynorm:
        samples.append(yscaler.inverse_transform(tsamples.cpu().detach().numpy()))
    else:
        samples.append(tsamples.cpu().detach().numpy())

    with open(args.output+'/samples.p', 'wb') as file_samples:
        pickle.dump(samples, file_samples)

    
    ##### Find MAP and write input_MAP.dat
    MAP = input_MAP(proposal, args.ynorm, yscaler, args.output, parnames, log)
                    
    ### Tidy up files  
    print('Tidying up...')
    # offsets = np.loadtxt(f'{args.output}/offsets_round_{r}_{0}.dat')
    # for i in range(1,args.processes):
    #     offsets=np.concatenate((offsets, np.loadtxt(f'{args.output}/offsets_round_{r}_{i}.dat')))
    # with open(f'{args.output}/offsets_round_{r}.npy', 'wb') as file_offsets:
    #     np.save(file_offsets, offsets)    
    
    Ts = np.load(f'{args.output}/T_round_{r}{0}.npy')
    for i in range(1,args.processes):
        Ts=np.concatenate((Ts, np.load(f'{args.output}/T_round_{r}{i}.npy')))
    with open(f'{args.output}/T_round_{r}.npy', 'wb') as file_Ts:
        np.save(file_Ts, Ts)  
    
    for j in range(args.processes):
        os.system(f'rm -rf {args.output}/offsets_round_{r}_{j}.dat')
        os.system(f'rm -rf {args.output}/T_round_{r}{j}.npy')

    r+=1
    
add_log('Inference ended.')

with open(args.output+'/post_equal_weights.txt', 'wb') as file_post_equal_weights:
    np.savetxt(file_post_equal_weights, samples[-1])

