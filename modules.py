from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding, PermutationInvariantEmbedding
import torch
from sbi import utils as utils
from scipy.stats.qmc import Sobol
import pickle
from sbi.inference import SNPE_C
from sbi.utils.get_nn_models import posterior_nn
import logging
import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn

def add_log(string):
    print(string)
    logging.info(string)
    return

def createEmbedding(embedding, embedding_type, embed_size, embed_hypers, rem_mean, nwvl):
    if embedding:
        if embedding_type=='FC':
            add_log('Using a fully connected embedding network.')
            summary = FCEmbedding(nwvl, embed_size)
        elif embedding_type=='CNN':
            add_log('Using a convolutional embedding network.')
            num_conv_layers, out_channels_per_layer, num_linear_layers, num_linear_units, kernel_size, output_dims = unroll_embed_hypers(embed_hypers, embed_size)
            if rem_mean:
                summary = CNNEmbedding(input_shape=(nwvl+10,), out_channels_per_layer=out_channels_per_layer, num_conv_layers=num_conv_layers, kernel_size=kernel_size,
                                   num_linear_layers=num_linear_layers, num_linear_units=num_linear_units, output_dim=output_dims[0])
            else:
                summary = CNNEmbedding(input_shape=(nwvl,), out_channels_per_layer=out_channels_per_layer, num_conv_layers=num_conv_layers, kernel_size=kernel_size,
                                   num_linear_layers=num_linear_layers, num_linear_units=num_linear_units, output_dim=output_dims[0])
        elif embedding_type=='multi':
            add_log('Using multiple embedding networks.')
            num_conv_layers, out_channels_per_layer, num_linear_layers, num_linear_units, kernel_size, output_dims = unroll_embed_hypers(embed_hypers, embed_size)
            summary = multiNet(nwvl,input_shape=(nwvl,), num_conv_layers=num_conv_layers, out_channels_per_layer=out_channels_per_layer, kernel_size=kernel_size,
                               num_linear_layers=num_linear_layers, num_linear_units=num_linear_units, output_dim=output_dims)
        else:
            raise TypeError('I have literally no clue what kind of embedding you want me to use.')
    else:
        summary = nn.Identity()
    
    return summary


def create_prior(prior_bounds, ynorm, yscaler, tail_bound, device, output):
    
    if ynorm:
        prior_min = torch.tensor(yscaler.transform(prior_bounds[:,0].reshape(1, -1)).reshape(-1))
        prior_max = torch.tensor(yscaler.transform(prior_bounds[:,1].reshape(1, -1)).reshape(-1))
    else:
        prior_min = torch.tensor(prior_bounds[:,0].reshape(1,-1))
        prior_max = torch.tensor(prior_bounds[:,1].reshape(1,-1))

    prior = utils.BoxUniform(low=prior_min.to(device, non_blocking=True), high=prior_max.to(device, non_blocking=True), device=device)
    
    return prior

def density_builder(flow, transforms, hidden, bins, summary, blocks, dropout, prior, device):

    density_estimator_build_fun = posterior_nn(model=flow, num_transforms=transforms, hidden_features=hidden, num_bins=bins, embedding_net=summary, num_blocks=blocks, dropout_probability=dropout)
    inference = SNPE_C(prior = prior, density_estimator=density_estimator_build_fun, device=device)
    
    return inference

def load_files(output, xnorm, ynorm, do_pca, num_rounds):
    add_log('Reading files from previous run...')
    
    np_theta = pickle.load(open(output+'/Y.p', 'rb'))
    arcis_spec = pickle.load(open(output+'/arcis_spec.p', 'rb'))
    r=len(arcis_spec.keys())
    posteriors=torch.load(output+'/posteriors.pt')
    proposal=posteriors[-1]
    samples=pickle.load(open(output+'/samples.p', 'rb'))
    try:
        logZs=pickle.load(open(output+'/evidence.p', 'rb'))
    except:
        add_log('oops')
        logZs=None
    inference=pickle.load(open(output+'/inference.p', 'rb'))
    inference_object=inference
    if xnorm:
        xscaler=pickle.load(open(output+'/xscaler.p', 'rb'))
    else:
        xscaler=None
    if ynorm:
        yscaler=pickle.load(open(output+'/yscaler.p', 'rb'))
    if do_pca:
        pca=pickle.load(open(output+'/pca.p', 'rb'))
    else:
        pca=None
    num_rounds+=r
    
    return np_theta, arcis_spec, posteriors, proposal, samples, logZs, inference, inference_object, xscaler, yscaler, pca, r, num_rounds

def sample_proposal(r, proposal, samples_per_round, prior_bounds, tail_bound, input2, n_global, init2):
    logging.info('Drawing '+str(samples_per_round)+' samples')
    print('Samples per round: ', samples_per_round)
    if r>0:
        theta = proposal.sample((samples_per_round,))
        np_theta = theta.cpu().detach().numpy().reshape([-1, len(prior_bounds)])
    elif r==0:
        try:
            np_theta = -tail_bound+(2*tail_bound)*Sobol(len(prior_bounds)).random_base2(int(np.log2(samples_per_round)))
        except:
            print('The number of samples per round needs to be a power of 2.')

    if input2!='aintnothinhere':
        for i in range(samples_per_round):
            if np_theta[i,n_global]<np_theta[i,init2]:
                lT = np_theta[i,n_global]
                hT = np_theta[i,init2]
                np_theta[i,n_global] = hT
                np_theta[i,init2] = lT
    
    return np_theta

def rot_int_cmj(w, s, vsini, eps=0.6, nr=10, ntheta=100, dif = 0.0):
    '''
    A routine to quickly rotationally broaden a spectrum in linear time.

    INPUTS:
    s - input spectrum

    w - wavelength scale of the input spectrum
    
    vsini (km/s) - projected rotational velocity
    
    OUTPUT:
    ns - a rotationally broadened spectrum on the wavelength scale w

    OPTIONAL INPUTS:
    eps (default = 0.6) - the coefficient of the limb darkening law
    
    nr (default = 10) - the number of radial bins on the projected disk
    
    ntheta (default = 100) - the number of azimuthal bins in the largest radial annulus
                            note: the number of bins at each r is int(r*ntheta) where r < 1
    
    dif (default = 0) - the differential rotation coefficient, applied according to the law
    Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)). Dif = .675 nicely reproduces the law 
    proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS. Dif = .23 is 
    similar to observed solar differential rotation. Note: the th in the above expression is 
    the stellar co-latitude, not the same as the integration variable used below. This is a 
    disk integration routine.

    '''

    ns = np.copy(s)*0.0
    tarea = 0.0
    dr = 1./nr
    for j in range(0, nr):
        r = dr/2.0 + j*dr
        area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - eps + eps*np.cos(np.arcsin(r)))
        for k in range(0,int(ntheta*r)):
            th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
            if dif != 0:
                vl = vsini * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
                ns += area * np.interp(w + w*vl/2.9979e5, w, s)
                tarea += area
            else:
                vl = r * vsini * np.sin(th)
                ns += area * np.interp(w + w*vl/2.9979e5, w, s)
                tarea += area
          
    return ns/tarea

def input_MAP(proposal, ynorm, yscaler, output, parnames, log):
    MAP = proposal.map(x=None, num_iter=100, num_to_optimize=100, learning_rate=0.01, init_method='posterior', 
                       num_init_samples=1000, save_best_every=10, show_progress_bars=True, force_update=False)

    if ynorm:
        MAP = yscaler.inverse_transform(MAP.reshape(1,-1))[0]
    
    with open(f'{output}/map_params.npy', 'wb') as map_params_file:
        np.save(map_params_file, MAP)

    os.system(f'cp {output}/input.dat {output}/input_map.dat')

    with open(f'{output}/input_map.dat', 'a') as mapfile:
        mapfile.write(f'\n')
        mapfile.write(f'******** MAP parameters ********\n')
        mapfile.write(f'\n')
        mapfile.write(f'makeai=.false.\n')
        mapfile.write(f'\n')
        for i in range(len(parnames)):
            if 'offset_' in parnames[i]:
                mapfile.write(f'*{parnames[i]}={MAP[i]}\n')
            else:
                if log[i]:
                    mapfile.write(f'{parnames[i]}={10**MAP[i]}\n')
                else:
                    mapfile.write(f'{parnames[i]}={MAP[i]}\n')  
    return MAP



