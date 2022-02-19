import argparse
import numpy as np
import math
import itertools
from itertools import combinations
    
def term_states(opponent):
    r = np.array([x for x in itertools.combinations([x for x in range(0,9)],4)])
    twos=np.sum(2*(10**r),axis=1)
    ones = []
    for indices in r:
        ones.append([x for x in range(0,9) if x not in indices])
    ones=np.sum(10**np.array(ones),axis=1)
    fullgrid = twos+ones

    # 1 wins when 2 plays and each player has plays thrice
    row_column = np.array([[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [6,4,2]])
    twos = np.sum(2*(10**row_column),axis=1)
    ones = np.sum((10**row_column),axis=1)
    winby1_1 = []
    for i in range(8):
        r = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in row_column[i,:]],3)])
        onesforwin1=np.sum(10**r,axis=1)
        winby1_1.append(onesforwin1 + twos[i])

    # 1 wins when 2 plays and each player has played 4 times
    winby1_2 = []
    twos1 = []
    for i in range(8):
        r1 = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in row_column[i,:]],1)])
        twos1.append(2*np.sum(10**r1,axis=1) + twos[i])
        for j in range(len(r1)):
            r2 = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in np.append(row_column[i,:],r1[j,:])],4)])
            onesforwin2=np.sum(10**r2,axis=1)
            win = (onesforwin2 + twos1[i][j])
            winby1_2.append(win)

    winby1 = np.append(np.reshape(np.array(winby1_1), (160,)),np.reshape(np.array(winby1_2), (240,)))

    winby2_1 = []
    for i in range(8):
        r = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in row_column[i,:]],2)])
        twosforwin1=2*np.sum(10**r,axis=1)
        winby2_1.append(twosforwin1 + ones[i])

    ones1 = []
    winby2_2 = []
    for i in range(8):
        r1 = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in row_column[i,:]],1)])
        ones1.append(1*np.sum(10**r1,axis=1) + ones[i])
        for j in range(len(r1)):
            r2 = np.array([x for x in itertools.combinations([x for x in range(0,9) if x not in np.append(row_column[i,:],r1[j,:])],3)])    
            twosforwin2 = 2*np.sum(10**r2,axis=1)
            win = (twosforwin2 + ones1[i][j])
            winby2_2.append(win)

    winby2 = np.append(np.reshape(np.array(winby2_1), (120,)),np.reshape(np.array(winby2_2), (480,)))
    # returns fullgrid terminal states, the winning states of the agent, and winning states of the opponent
    if opponent == 2:
        return fullgrid, winby1, winby2
    if opponent == 1:
        return fullgrid, winby2, winby1
    else:
        return None
        
def zeros(x):
    l = np.array([int(m) for m in str(x)])
    if len(l) < 9:
        return np.append(np.array([i for i in range(9-len(l))]), np.where(l == 0)[0] + (9-len(l))).astype(int)
    else:
        return np.where(l == 0)[0]
        
def winner(state, row_column, opponent):
    state = np.array([int(x) for x in str(state)])
    opp_positions = np.where(state == opponent)[0]
    for i in range(np.shape(row_column)[0]):
        s=0
        for k in range(3):
            if row_column[i,k] in opp_positions:
                s+=1
            if s==3:
                return True
    return False    
   
def t_r_grid(ag_states, opp_states, opp_prob, opponent, discount):
    termstates, wins, losses = term_states(opponent)
    mdp = dict()
    mdp["states"] = len(ag_states)+len(termstates)+len(wins)+len(losses)
    mdp["end"] = range(len(ag_states), mdp["states"])
    mdp["actions"] = 9
    mdp["r"] = np.zeros((mdp["states"], mdp["actions"], mdp["states"]))
    mdp["t"] = np.zeros((mdp["states"], mdp["actions"], mdp["states"]))
    mdp["discount"] = discount
    row_column = np.array([[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [6,4,2]])
    if opponent == 2:
        for state_id in range(len(ag_states)):
            state = int(ag_states[state_id])
            actions = zeros(state)
            for action in actions:
                s1 = state + (10**(8 - action))
                if len(zeros(s1)) == 0:
                    s1_id = int(len(ag_states) + np.where(termstates == s1)[0][0])
                    mdp["t"][state_id, action, s1_id] = 1

                else:
                    s1_present = np.where(opp_states == s1)[0]

                    if len(s1_present) == 0:
                        s1_id = int(len(ag_states) + len(termstates) + len(wins)+ np.where(losses==s1)[0][0])
                        mdp["t"][state_id, action, s1_id] = 1
                    else:
                        i = s1_present[0]
                        for j in range(len(opp_prob[i,:])):
                            if opp_prob[i,j] != 0:
                                s2 = s1 + (2*(10**(8 - j)))
                                s2_present = np.where(ag_states == s2)[0]
                                if len(s2_present) == 0: 
                                    s2_id = int(len(ag_states) + len(termstates) + np.where(wins==s2)[0][0])
                                    mdp["t"][state_id, action, s2_id] = opp_prob[i,j]
                                    mdp["r"][state_id, action, s2_id] = 1
                                else:
                                    s2_id = int(np.where(ag_states == s2)[0][0])
                                    mdp["t"][state_id,action,s2_id] = opp_prob[i,j]
            invalidactions = [x for x in range(9) if x not in actions]   
            for invaction in invalidactions:
                mdp["t"][state_id, int(invaction), state_id] = 1
                mdp["r"][state_id, int(invaction), state_id] = -100

            
    if opponent == 1:
        for state_id in range(len(ag_states)):
            state = int(ag_states[state_id])
            actions = zeros(state)
            for action in actions:
                s1 = state + (2*(10**(8 - action)))
                s1_present = np.where(opp_states == s1)[0]
                if len(s1_present) == 0:
                    s1_id = int(len(ag_states) + len(termstates) + len(wins)+ np.where(losses==s1)[0][0])
                    mdp["t"][state_id, action, s1_id] = 1
                else:
                    i = s1_present[0]
                    for j in range(len(opp_prob[i,:])):
                        if opp_prob[i,j] != 0:
                            s2 = s1 + (1*(10**(8 - j)))
                      
                            if len(zeros(s2)) == 0:
                                s2_id = int(len(ag_states) + np.where(termstates == s2)[0][0])
                                mdp["t"][state_id, action, s2_id] = 1
                                if winner(s2, row_column, opponent):
                                    mdp["r"][state_id,action,s2_id] = 1
                            else:
                                s2_present = np.where(ag_states == s2)[0]
                                
                                if len(s2_present) == 0:
                                    s2_id = int(len(ag_states) + len(termstates) + np.where(wins==s2)[0][0])
                                    mdp["t"][state_id, action, s2_id] = opp_prob[i,j]
                                    mdp["r"][state_id, action, s2_id] = 1
                                else:
                                    s2_id = int(np.where(ag_states == s2)[0][0])
                                    mdp["t"][state_id,action,s2_id] = opp_prob[i,j]
            invalidactions = [x for x in range(9) if x not in actions]
            for invaction in invalidactions:
                mdp["t"][state_id, invaction, state_id] = 1
                mdp["r"][state_id, invaction, state_id] = -100
    return mdp

def write_to_output(mdp):
    out = ""
    out+="numStates "+str(mdp["states"])+'\n'
    out+="numActions "+str(mdp["actions"])+'\n'
    out+="end "
    end = ""
    for endstate in mdp["end"]:
        end+=str(endstate)+" "
    out+= end.rstrip()+'\n'
    transitions = ""
    for i in range(mdp["states"] - len(mdp["end"])):
        for j in range(mdp["actions"]):
            for k in range(mdp["states"]):
                if (mdp["t"][i,j,k] != 0):
                    transitions += "transition "+str(i)+" "+str(j)+" "+str(k)+" "+str(mdp["r"][i,j,k])+" "+str(mdp["t"][i,j,k])+'\n'
    out+= transitions
    out+="mdptype episodic"+'\n'
    out+="discount "+str(mdp["discount"])
    print(out)

def input_opp_policy(policypath):
    with open(policypath) as instance1:
        opp_params = instance1.readlines()

    oppstates = np.empty((len(opp_params)-1,))
    opp_prob = np.empty((len(opp_params)-1,9))
    opponent = int(opp_params[0].rstrip())
    
    for i in range(1,len(opp_params)):
        param = np.array(opp_params[i].rstrip().split(" "))
        oppstates[i-1] = int(param[0])
        opp_prob[i-1,:] = param[1:].astype(float)
    
    return oppstates, opp_prob, opponent


def input_states(statespath):
    with open(statespath) as instance2:
        ag_params = instance2.readlines()

    agstates = np.empty((len(ag_params),))    
    for i in range(0,len(ag_params)):
        param = ag_params[i].rstrip()
        agstates[i] = int(param)
    return agstates

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()

    parser.add_argument('--policy', type = str, help = 'path/to/policy/of/opponent')
    parser.add_argument('--states', type = str, help = 'path/to/valid/states/of/agent')

    args = parser.parse_args()

    policypath = args.policy
    statespath = args.states

    oppstates, opp_prob, opponent = input_opp_policy(policypath)
    agstates = input_states(statespath)
    mdp = t_r_grid(agstates, oppstates, opp_prob, opponent, 1)
    write_to_output(mdp)

