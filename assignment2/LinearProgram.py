from pulp import*
import numpy as np

def LinearProgram(mdp):
    problem = LpProblem("MDPPlanning", LpMinimize)
    valuefs = np.array(list(LpVariable.dicts("V", range(mdp["states"])).values()))
    
    # Objective Function
    problem += lpSum([valuefs[i] for i in range(valuefs.shape[0])]) 
    
    # Constraints:
    constraints = np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*valuefs), axis=2)
    for i in range(mdp["states"]):
        for j in range(mdp["actions"]):
            problem += valuefs[i] >= constraints[i][j]
          
    problem.solve(apis.PULP_CBC_CMD(msg=0))
    valuefs_soln = np.array([valuefs[i].value() for i in range(valuefs.shape[0])])
    
    policy = np.argmax(np.sum(mdp["t"]*(mdp["r"] + mdp["discount"]*valuefs_soln), axis=2), axis=1)
    
    return valuefs_soln, policy