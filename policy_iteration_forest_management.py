
##########################################
# IMPORTS
##########################################
import numpy as np
import random as rd
from assignement_4.Forest_management_mdp import *
import pickle
import time

##########################################
# POLICY ITERATION
##########################################
def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (s1,p) in mdp.T(s['number'], a,mode='random')])

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmax(seq, fn):
    return argmin(seq, lambda x: -fn(x))


##########################################
# WITH METRICS
##########################################


def policy_improvement_forest_management_metrics(mdp:ForestManagement(),signature,epsilon=0.0001):
    pi = np.random.randint(2, size=mdp.number_of_states)
    U = np.zeros(mdp.number_of_states+1)
    gamma = mdp.gamma
    U[-1] = mdp.reward_wait/mdp.gamma #Equivalent to giving value of 0 and reward of reaching final state
    action = {'cut':0, 'wait':1}
    iterations = []
    policy_changes_iterations =[]
    it=0
    p = mdp.fire_prob
    number_policy_wait = 0
    for k in range(pi.shape[0]):
        if pi[k] == 1:
            number_policy_wait+=1
    number_policy_wait_iterations = []
    delta_iterations = []
    start = time.time()
    c=0
    while True:
        start = time.time()
        U = policy_evaluation_forest_management_metrics(pi, U, mdp,epsilon)
        unchanged = True
        for s in range(mdp.number_of_states):
            equivalent=False
            value_cut = mdp.reward_cut
            value_wait = (1-mdp.fire_prob)*mdp.gamma*U[s+1]
            if value_wait > value_cut:
                best_action = action['wait']
            elif value_wait < value_cut:
                best_action = action['cut']
            else:
                best_action = rd.randint(0,1)
                equivalent = True


            if best_action != pi[s]:
                if best_action == 1:
                    number_policy_wait+=1
                else:
                    number_policy_wait-=1
                pi[s] = best_action
                if equivalent == False:
                    unchanged = False
        end = time.time()
        if(c <4):
            c+=1
            print(mdp.number_of_states)
            print(end-start)
        number_policy_wait_iterations.append(number_policy_wait)
        if unchanged:
            print("total time :"+str(time.time() - start))
            pickle.dump(number_policy_wait_iterations,open("assignement_4/forest_management/"+signature+"number_policy_wait_iterations"+"_gamma_"+str(gamma)+"_fire_prob_"+str(p)+".p","wb"))

            return U,pi

def policy_evaluation_forest_management_metrics(pi, U, mdp:ForestManagement(),epsilon=0.0001):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    action = {'cut':0,'wait':1}
    while True:
        delta = 0
        U1 = U.copy()
        for s in range(mdp.number_of_states):
            if pi[s] == action['cut']:
                value = mdp.reward_cut
            else:
                value = (1-mdp.fire_prob)*mdp.gamma*U1[s+1]
            U[s] = value
            delta = max(abs(U[s]-U1[s]),delta)
        if delta <= epsilon * (1 - gamma) / gamma:
            return U


##########################################
# WITHOUT METRICS
##########################################

def policy_improvement_forest_management(mdp:ForestManagement(),epsilon=0.001):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    pi = np.zeros(mdp.number_of_states)
    U = np.zeros(mdp.number_of_states+1)
    U[-1] = mdp.reward_wait/mdp.gamma #Equivalent to giving value of 0 and reward of reaching final state
    action = {'cut':0, 'wait':1}
    while True:
        U = policy_evaluation_forest_management(pi, U, mdp,epsilon)
        unchanged = True
        for s in range(mdp.number_of_states):
            equivalent=False
            value_cut = mdp.reward_cut
            value_wait = (1-mdp.fire_prob)*mdp.gamma*U[s+1]
            if value_wait > value_cut:
                best_action = action['wait']
            elif value_wait < value_cut:
                best_action = action['cut']
            else:
                best_action = rd.randint(0,1)
                equivalent = True


            if best_action != pi[s]:
                pi[s] = best_action
                if equivalent == False:
                    unchanged = False
        if unchanged:
            return U,pi

def policy_evaluation_forest_management(pi, U, mdp:ForestManagement(),epsilon=0.001):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    action = {'cut':0,'wait':1}
    while True:
        delta = 0
        U1 = U.copy()
        for s in range(mdp.number_of_states):
            if pi[s] == action['cut']:
                value = mdp.reward_cut
            else:
                value = (1-mdp.fire_prob)*mdp.gamma*U1[s+1]
            U[s] = value
            delta = max(abs(U[s]-U1[s]),delta)
        if delta <= epsilon * (1 - gamma) / gamma:
            return U