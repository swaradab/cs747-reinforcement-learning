import argparse
import numpy as np

def decode_policy(valpolicypath, statespath, agent):
    opponent = 2 if agent == 1 else 1

    with open(statespath) as instance1:
        ag_params = instance1.readlines()

    agstates = np.empty((len(ag_params),))    
    for i in range(0,len(ag_params)):
        param = ag_params[i].rstrip()
        agstates[i] = np.array(int(param))
        
    policy = np.zeros((len(agstates), 9))

    with open(valpolicypath) as instance2:
        policy_params = instance2.readlines()
        
    for i in range(len(agstates)):
       param = policy_params[i].strip().split(" ") 
       action = int(param[1])
       policy[i,action] = 1
     
    out = str(agent)+'\n'
    for i in range(len(agstates)):
        l = [int(m) for m in str(int(agstates[i]))]
        if len(l) == 9:
            out += str(int(agstates[i]))
        else:
            for j in range(9-len(l)):
                out+="0"
            out += str(int(agstates[i]))
        for j in range(9):
            out += " "+str(int(policy[i,j]))
        out += '\n'
        
    print(out.rstrip())

 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument('--value-policy', type = str, help = 'path/to/value-policy/file/of/agent')
    parser.add_argument('--states', type = str, help = 'path/to/valid/states/of/agent')
    parser.add_argument('--player-id', type = str, help = 'id/of/agent')

    args = parser.parse_args()
    valpolicypath = args.value_policy
    statespath = args.states
    agent = int(args.player_id)
    decode_policy(valpolicypath, statespath, agent)

