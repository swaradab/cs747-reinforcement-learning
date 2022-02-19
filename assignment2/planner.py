import argparse
import numpy as np
from valueIteration import valueIteration
from LinearProgram import LinearProgram
from HowardPolicyIteration import HowardPolicyIteration
import time

def printOutput(valuefs, policy):
    out = ""
    for i in range(valuefs.shape[0]):
        out += "{:.6f}".format(valuefs[i]) + " " + str(policy[i]) + '\n'   
    print(out.rstrip())
   
def input_mdp(path):
    with open(path) as instance:
        params = instance.readlines()

    mdp = dict()
    mdp["states"] = int(params[0].rstrip().split(" ")[1])
    mdp["actions"] = int(params[1].rstrip().split(" ")[1])
    mdp["r"] = np.zeros((mdp["states"], mdp["actions"], mdp["states"]))
    mdp["t"] = np.zeros((mdp["states"], mdp["actions"], mdp["states"]))

    for i in range(3,len(params)):
        param = np.array(params[i].rstrip().split(" "))
        if param[0] == 'transition':
            mdp["r"][int(param[1]),int(param[2]),int(param[3])] = float(param[4])
            mdp["t"][int(param[1]),int(param[2]),int(param[3])] = float(param[5])
        
        if param[0] == 'mdptype':
            if param[-1] == 'episodic':
                mdp["end"] = int(params[2].rstrip().split(" ")[1])
            else:
                mdp["end"] = -1
        if param[0] == 'discount':
            mdp["discount"] = float(param[-1])
    return mdp
     
def mdp_planner(mdp, algo):
    if algo == None:
        valuefs, policy = valueIteration(mdp)
        printOutput(valuefs, policy)

    else: 
      
        if algo == 'vi':
            valuefs, policy = valueIteration(mdp)
            printOutput(valuefs, policy)
            
        if algo == 'lp':
            valuefs, policy = LinearProgram(mdp)
            printOutput(valuefs, policy)
            
        if algo == 'hpi':
            valuefs, policy = HowardPolicyIteration(mdp)
            printOutput(valuefs, policy)

if __name__ == "__main__":            
    parser = argparse.ArgumentParser()

    parser.add_argument('--mdp', type = str, help = 'path/to/mdp/instance/')
    parser.add_argument('--algorithm', type = str, help = 'name of algorithm for mdp planning')

    args = parser.parse_args()

    path = args.mdp
    algo = args.algorithm

    mdp = input_mdp(path)
    mdp_planner(mdp, algo)