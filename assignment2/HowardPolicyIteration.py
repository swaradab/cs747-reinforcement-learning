import numpy as np
np.random.seed(10)

def policy_eval(mdp, policy):
    vfs = np.zeros((mdp["states"],))
    margin = 1e-8
    err = 1
    while (err > margin):
        q = np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*vfs), axis=2)
        v = np.copy(vfs)
        for i in range(policy.shape[0]):
            vfs[i] = q[i][policy[i]]
        err = np.amax(np.absolute(vfs-v))
    return vfs

def HowardPolicyIteration(mdp):
    policy = np.random.randint(0, mdp["actions"], mdp["states"])
 
    valuefs = policy_eval(mdp, policy)
    qvalues = np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*valuefs), axis=2)
    improvactions = np.greater(qvalues, np.hstack([np.reshape(valuefs, (mdp["states"],1)) for i in range(mdp["actions"])])).astype(int)
    for i in range(mdp["states"]):
            improvactions[i][policy[i]] = 0
    improvstates = np.sum(np.sum(improvactions > 0, axis=1) > 0)
    
    while(improvstates > 0):
        for i in range(mdp["states"]):
            if np.sum(improvactions[i] > 0) > 0:
                policy[i] = np.argmax(improvactions[i])
        valuefs = policy_eval(mdp, policy)
        qvalues = np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*valuefs), axis=2)
       
        improvactions = np.greater(qvalues, np.hstack([np.reshape(valuefs, (mdp["states"],1)) for i in range(mdp["actions"])])).astype(int)
        for i in range(mdp["states"]):
            improvactions[i][policy[i]] = 0
        improvstates = np.sum(np.sum(improvactions > 0, axis=1) > 0)
  
    return valuefs, policy