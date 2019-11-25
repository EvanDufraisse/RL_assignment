'''
Suppose the state manager as a list of states with their actions and current Q_values in form a list in fourth position (index 3)
[[action,qvalue],[action,qvalue]...]
'''
import numpy as np
import random as rd
from assignement_4.Forest_management_mdp import *

def Q_learning_Forest_Management(mdp:ForestManagement(),step_size=0.1,epsilon =0.1,n_episodes=500,alpha=0.1,exploration=1):
    exp_methods = {'UCB':0,"eps_greedy":1}
    Q = dict([(s, np.zeros(2)) for s in range(mdp.number_of_states)])
    if(exp_methods['UCB'] == exploration):
        Q_choice = dict([(s, np.ones(2)) for s in range(mdp.number_of_states)])
        exploration = UCB_exploration_forest_management
    else:
        exploration = epsilon_greedy_exploration_forest_management
        Q_choice = None
    gamma = mdp.gamma
    action = {'cut':0,'wait':1}
    number_of_states_wait = 0
    number_of_states_wait_list = []
    for episode in range(n_episodes):
        s = 0
        unfinished = True
        while unfinished:
            agent_action = exploration(Q,Q_choice,epsilon,s)
            if agent_action == action['cut']:
                unfinished = False
                Q[s][agent_action] = Q[s][agent_action] + alpha*(mdp.reward_cut  - Q[s][agent_action])
            else:
                if s == mdp.number_of_states-1:
                    Q[s][agent_action] = Q[s][agent_action] + alpha*(mdp.reward_wait - Q[s][agent_action])
                    unfinished = False
                else:
                    if np.random.uniform() <= mdp.fire_prob: # Fire burns the forest
                        Q[s][agent_action] = Q[s][agent_action] - alpha*Q[s][agent_action]
                        unfinished = False
                    else: # Fire hasn't burn the forest
                        Q[s][agent_action] = Q[s][agent_action] + alpha*(gamma*Q[s+1].max() - Q[s][agent_action])
                        s=s+1

    return Q
            
def epsilon_greedy_exploration_forest_management(Q,Q_choice, epsilon, s):
    if(Q[s][0]==Q[s][1]):
        return rd.randint(0,1)
    else:
        if np.random.uniform()<=epsilon:
            return np.argmin(Q[s])
        else:
            return np.argmax(Q[s])


def UCB_exploration_forest_management(Q,Q_choice,epsilon,s):
    beta = 10
    Q_val_cut = Q[s][0] + beta*np.sqrt(np.log(Q_choice[s].sum())/Q_choice[s][0])
    Q_val_wait = Q[s][1] + beta*np.sqrt(np.log(Q_choice[s].sum())/Q_choice[s][1])
    if(Q_val_cut==Q_val_wait):
        action_chosen = rd.randint(0,1)
        Q_choice[s][action_chosen]+=1
        return action_chosen
    elif Q_val_cut > Q_val_wait:
        action_chosen = 0
        Q_choice[s][action_chosen]+=1
        return action_chosen
    else:
        Q_choice[s][1]+=1
        return 1


