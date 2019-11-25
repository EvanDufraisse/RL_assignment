from assignement_4.Forest_management_mdp import *
import random as rd
import pickle
import time
def value_iteration_forest_management(mdp:ForestManagement(), epsilon=0.0001):
    "Solving an MDP by value iteration."
    U1 = np.zeros(mdp.number_of_states+1)
    U1[-1] = mdp.reward_wait/mdp.gamma
    policy = np.zeros(mdp.number_of_states)
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    action = {'cut':0,'wait':1}
    while True:
        U = U1.copy()
        delta = 0
        for s in range(mdp.number_of_states):
            value_cut = mdp.reward_cut
            value_wait = (1-mdp.fire_prob)*mdp.gamma*U[s+1]
            if value_wait < value_cut:
                policy[s] = action['cut']
                U1[s] = value_cut
            elif value_wait > value_cut:
                policy[s] = action['wait']
                U1[s] = value_wait
            else:
                chosen_action = rd.randint(0,1)
                policy[s] = chosen_action
                if chosen_action == 0:
                    U1[s] = value_cut
                else:
                    U1[s] = value_wait
            delta = max(delta,abs(U1[s]-U[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
                return U,policy


def value_iteration_forest_management_metrics(mdp:ForestManagement(),signature,epsilon=0.0001):
    "Solving an MDP by value iteration."
    U1 = np.zeros(mdp.number_of_states+1)
    U1[-1] = mdp.reward_wait/mdp.gamma
    policy = np.zeros(mdp.number_of_states)
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    action = {'cut':0,'wait':1}
    iterations = []
    policy_changes_iterations =[]
    it=0
    c=0
    p = mdp.fire_prob
    number_policy_wait = 0
    number_policy_wait_iterations = []
    delta_iterations = []
    while True:
        start = time.time()
        U = U1.copy()
        delta = 0
        policy_change =0
        for s in range(mdp.number_of_states):
            value_cut = mdp.reward_cut
            value_wait = (1-mdp.fire_prob)*mdp.gamma*U[s+1]
            old_policy = policy[s]
            if value_wait < value_cut:
                policy[s] = action['cut']
                if policy[s] != old_policy:
                    policy_change+=1
                    number_policy_wait-=1
                U1[s] = value_cut
            elif value_wait > value_cut:
                policy[s] = action['wait']
                if policy[s] != old_policy:
                    policy_change+=1
                    number_policy_wait+=1
                U1[s] = value_wait
            else:
                chosen_action = rd.randint(0,1)
                policy[s] = chosen_action
                if chosen_action == 0:
                    U1[s] = value_cut
                else:
                    U1[s] = value_wait
                if policy[s] != old_policy:
                    policy_change+=1
                    number_policy_wait+=1
            
            delta = max(delta,abs(U1[s]-U[s]))
        end = time.time()
        if(c <4):
            c+=1
            print(mdp.number_of_states)
            print(end-start)
        iterations.append(it)
        policy_changes_iterations.append(policy_change)
        number_policy_wait_iterations.append(number_policy_wait)
        delta_iterations.append(delta)
        #print("delta: " + str(delta) )
        if delta <= epsilon * (1 - gamma) / gamma:
            names = ["Policy","Value","policy_changes_iterations","number_policy_wait_iterations","delta_iterations",'iterations']
            variables = [policy,U,policy_changes_iterations,number_policy_wait_iterations,delta_iterations,iterations]
            #print(delta_iterations)
            for k in range(len(names)):
                pickle.dump(variables[k],open("assignement_4/forest_management/value_iteration/"+signature+names[k]+"_gamma_"+str(gamma)+"_fire_prob_"+str(p)+".p","wb"))
            return U,policy