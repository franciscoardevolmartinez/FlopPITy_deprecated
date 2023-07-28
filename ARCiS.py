import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def read_arcis_input(inputfile, twoterms):
    inp = []
    with open(inputfile, 'rb') as arcis_input:
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
        #I think this is just checking the number of atm layers to initialize PT arrays? Sounds inefficient and I should change it
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
            elif twoterms:
                if clean_in[i][16:-1]=='Rp' or clean_in[i][16:-1]=='loggP':
                    parnames.append(clean_in[i][16:-1])
                    if clean_in[i+3]=='fitpar:log=.true.':
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                    elif clean_in[i+3]=='fitpar:log=.false.':
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                else:
                    parnames.append(clean_in[i][16:-1]+'_1')
                    parnames.append(clean_in[i][16:-1]+'_2')
                    if clean_in[i+3]=='fitpar:log=.true.':
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                        prior_bounds.append([np.log10(float(clean_in[i+1][11:].replace('d','e'))), np.log10(float(clean_in[i+2][11:].replace('d','e')))])
                    elif clean_in[i+3]=='fitpar:log=.false.':
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                        prior_bounds.append([float(clean_in[i+1][11:].replace('d','e')), float(clean_in[i+2][11:].replace('d','e'))])
                i+=4
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
          
    return parnames, prior_bounds,obs, obs_spec,noise_spec, nr