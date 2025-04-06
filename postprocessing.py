'''
Here user-defined post-processing options can be added. Very basically, these should be written as functions 
that take a spectrum and some parameters and return a modified spectrum.

Example:

def postprocess(spectrum, *parameters):
    new_spec = do something
    return new_spec
'''

def scaling(spec, scale):
    for j in range(1,n_obs):
        arcis_spec[i][int(sum(nwvl[:j])):int(sum(nwvl[:j+1]))] *= scaling[i][j-1]