import numpy as np
from encoder import input_states, zeros, t_r_grid
from valueIteration import valueIteration

def printPolicy(policy, agstates, player, initialisation, rs, iteration, filepath):
    out = initialisation+" initialisation, "+"random seed "+str(rs)+", iteration "+str(iteration)+'\n'
    for i in range(len(agstates)):
        l = [int(m) for m in str(int(agstates[i]))]
        if len(l) == 9:
            out += str(int(agstates[i]))
        else:
            for j in range(9-len(l)):
                out+="0"
            out += str(int(agstates[i]))
        for j in range(9):
            out += " "+str(policy[i,j])
        out += '\n'
    
    output = open(filepath,"a")
    output.write(out)
    output.close()   

def initialise(itype, p2states, p2states_no, rs):
    np.random.seed(rs)
    p2policy = np.zeros((p2states_no, 9))
    if itype == "uniform":
        for state in range(p2states_no):
            p2state = int(p2states[state])
            actions = zeros(p2state)
            for action in actions:
                p2policy[state, action] = (1/len(actions))
                
    if itype == "random":
        for state in range(p2states_no):
            p2state = int(p2states[state])
            actions = zeros(p2state)
            sumofprobs = 0
            for i in range(len(actions)-1):
                action = actions[i]
                p2policy[state, action] = np.random.randint(0,1001 - sumofprobs)
                sumofprobs += p2policy[state, action]
            p2policy[state, actions[-1]] = 1000 - sumofprobs
        p2policy = p2policy/1000
        
    if itype == "deterministic":
        for state in range(p2states_no):
            p2state = int(p2states[state])
            actions = zeros(p2state)
            action = np.random.choice(actions)
            p2policy[state, action] = 1
                       
    return p2policy

def policy_to_prob(policy, states_no):
    prob = np.zeros((states_no,9))
    for i in range(states_no):
        prob[i, int(policy[i])] = 1
    return prob
    
def prob_to_policy(prob, states_no):
    policy = np.zeros((states_no,))
    for i in range(states_no):
        policy[i] = np.where(prob[i,:]==1)[0][0]
    return policy
    
def perform_iterations_p2_first(initialisation, statesp1path, statesp2path, rs):
    p2states = input_states(statesp2path)
    p2states_no = len(p2states)
    p1states = input_states(statesp1path)
    p1states_no = len(p1states)
    
    p1_policies = []
    p2_policies = []
    
    p2prob = initialise(initialisation, p2states, p2states_no, rs)
    #print(p2prob)
    p2_policies.append(p2prob)
    #print(p2_policies[0])

    mdp_p1 = t_r_grid(p1states, p2states, p2prob, 2, 1)
    p1valuefs, p1policy = valueIteration(mdp_p1)
    p1prob = policy_to_prob(p1policy, p1states_no)
    p1_policies.append(p1prob)
    out = initialisation+" initialisation, "+"random seed "+str(rs)+'\n'
    for i in range(1,10):
        mdp_p2 = t_r_grid(p2states, p1states, p1prob, 1, 1)
        p2valuefs, p2policy = valueIteration(mdp_p2)
        p2prob = policy_to_prob(p2policy, p2states_no)
        p2_policies.append(p2prob)
        
        mdp_p1 = t_r_grid(p1states, p2states, p2prob, 2, 1)
        p1valuefs, p1policy = valueIteration(mdp_p1)
        p1prob = policy_to_prob(p1policy, p1states_no)
        p1_policies.append(p1prob)
               
        p1Policydiff = np.linalg.norm(p1_policies[i] - p1_policies[i-1])
        p2Policydiff = np.linalg.norm(p2_policies[i] - p2_policies[i-1])
        out+="Frobenius norm of of p1(iteration "+str(i-1)+") - p1(iteration "+str(i)+"): "+str(p1Policydiff)+'\n'
        out+="Frobenius norm of of p2(iteration "+str(i-1)+") - p2(iteration "+str(i)+"): "+str(p2Policydiff)+'\n'
    out+='\n'
    return out, p2_policies, p1_policies
    
def perform_iterations_p1_first(initialisation, statesp1path, statesp2path, rs):
    p2states = input_states(statesp2path)
    p2states_no = len(p2states)
    p1states = input_states(statesp1path)
    p1states_no = len(p1states)
    
    p1_policies = []
    p2_policies = []
    
    p1prob = initialise(initialisation, p1states, p1states_no, rs)
    #print(p2prob)
    p1_policies.append(p1prob)
    #print(p2_policies[0])

    mdp_p2 = t_r_grid(p2states, p1states, p1prob, 1, 1)
    p2valuefs, p2policy = valueIteration(mdp_p2)
    p2prob = policy_to_prob(p2policy, p2states_no)
    p2_policies.append(p2prob)
    out = initialisation+" initialisation, "+"random seed "+str(rs)+'\n'
    for i in range(1,10):
        mdp_p1 = t_r_grid(p1states, p2states, p2prob, 2, 1)
        p1valuefs, p1policy = valueIteration(mdp_p1)
        p1prob = policy_to_prob(p1policy, p1states_no)
        p1_policies.append(p1prob)
        
        mdp_p2 = t_r_grid(p2states, p1states, p1prob, 1, 1)
        p2valuefs, p2policy = valueIteration(mdp_p2)
        p2prob = policy_to_prob(p2policy, p2states_no)
        p2_policies.append(p2prob)
               
        p1Policydiff = np.linalg.norm(p1_policies[i] - p1_policies[i-1])
        p2Policydiff = np.linalg.norm(p2_policies[i] - p2_policies[i-1])
        out+="Frobenius norm of of p1(iteration "+str(i-1)+") - p1(iteration "+str(i)+"): "+str(p1Policydiff)+'\n'
        out+="Frobenius norm of of p2(iteration "+str(i-1)+") - p2(iteration "+str(i)+"): "+str(p2Policydiff)+'\n'
    out+='\n'
    return out, p2_policies, p1_policies
    
if __name__ == "__main__": 

    statesp1path = "../submission/data/attt/states/states_file_p1.txt"
    statesp2path = "../submission/data/attt/states/states_file_p2.txt"
    
    # to check convergence for different iterations and random states
    for initialisation in ["uniform", "deterministic", "random"]:
        for rs in [1, 5, 10, 50, 100]:
            out, _, _ = perform_iterations_p2_first(initialisation, statesp1path, statesp2path, rs)
            output = open("/host/submission/convergence.txt","a")
            output.write(out)
            output.close()
            
    # to print out policies generated of a single iterations cycle
    out, p2policies, p1policies = perform_iterations_p2_first("uniform", statesp1path, statesp2path, 5)
    
    for i in range(len(p1policies)):
        p2policy = p2policies[i]
        p1policy = p1policies[i]
        printPolicy(p2policy, input_states(statesp2path), 2, "uniform", 5, i, "/host/submission/p2policyiter"+str(i)+"random.txt")
        printPolicy(p1policy, input_states(statesp1path), 1, "uniform", 5, i, "/host/submission/p1policyiter"+str(i)+"random.txt")

    
        