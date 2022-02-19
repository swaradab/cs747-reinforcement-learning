import numpy as np

def valueIteration(mdp):
    valuefs = np.zeros((1, mdp["states"]))
    prev = np.zeros((mdp["states"],1))
    policy = np.zeros((mdp["states"],1))
    while(1):
        result = np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*valuefs), axis=2)
        valuefs_new = np.amax(result, axis=1)
        policy = np.argmax(result, axis=1)
        if np.amax(np.absolute(valuefs_new-valuefs)) == 0:
            valuefs = valuefs_new
            break
        valuefs = valuefs_new
        
    return valuefs, policy
