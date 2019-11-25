###################################################################
# IMPORTS
###################################################################

from assignement_4.Tictactoe_mdp import *
import pickle
import numpy as np
import random as rd
import time


###################################################################
# WITH METRICS
###################################################################
def Q_learning_metrics(mdp:Tictactoe(),mode,signature,step_performance_list,epsilon =0.1,n_episodes=500,alpha=0.1,big=False):
    gamma = mdp.gamma
    for key,val in mdp.states_manager.states.items():
        val['Q'] = np.zeros(len(val['actions']))
        if(len(val['actions'])==0):
            val['Q'] = np.zeros(1)
        val['actions_explored'] = np.ones(len(val['actions']))
        val['explored'] = 0
        if not val['terminal']:
            val['policy'] = np.random.choice(val['actions'])
    score_random = []
    score_intelligent = []
    step_iteration = []
    
    time_step_iteration = []
    s_init = mdp.states_manager.get_state(0)
    Q_values_init_iterations = [s_init['Q']]
    unexplored = len(mdp.states_manager.states)-1
    unexplored_iterations = [unexplored]
    print("problem initialized")
    start = time.time()
    k=1
    step_performance = step_performance_list[k]
    for episode in range(n_episodes):
        if(not big):
            Q_values_init_iterations.append(s_init['Q'])

        if(episode == step_performance):
            step = step_performance_list[k] - step_performance_list[k-1]
            time_elapsed = time.time() - start
            print("episode : "+str(episode))
            print("time last it : "+str(time_elapsed))
            time_step_iteration.append(time_elapsed)            
            current_time = time.time()
            if(big):
                Q_values_init_iterations.append(s_init['Q'])
                unexplored_iterations.append(unexplored)
            #update_policy(mdp)
            scores = performance_measure_dict(mdp,repeat=100000)
            score_random.append(scores[0])
            score_intelligent.append(scores[1])
            step_iteration.append(episode)
            print("time performance step : "+str(time.time()-current_time))
            k=k+1
            if(k<len(step_performance_list)):
                step_performance = step_performance_list[k]
            start = time.time() - time_elapsed

        state = mdp.states_manager.get_state(0)
        state['explored']+=1
        while not state['terminal']:
            chosen_action_index = epsilon_greedy_exploration_metrics(epsilon,state)
            agent_action = state['actions'][chosen_action_index]
            state['actions_explored'][chosen_action_index] +=1
            potential_actions = mdp.T(state['number'],agent_action,state['actions'],mode=mode)
            final_state_number = mdp.choose_action(potential_actions)
            final_state = mdp.states_manager.get_state(final_state_number)
            state['Q'][chosen_action_index] = state['Q'][chosen_action_index] + alpha*(final_state['reward']+gamma*(final_state['Q'].max()) -state['Q'][chosen_action_index])
            state['policy'] = state['actions'][np.argmax(state['Q'])]
            state = final_state
            state['explored'] +=1
            if(state['explored']==1):
                unexplored-=1
        if(not big):
            unexplored_iterations.append(unexplored)
    
    policy = create_policy(mdp)
    exploration_stats = exploration_statitics(mdp)
    variables = [score_random,score_intelligent,step_iteration,time_step_iteration,
    Q_values_init_iterations,unexplored_iterations,policy,exploration_stats]
    names = ["score_random","score_intelligent","step_iteration","time_step_iteration",
    "Q_values_init_iterations","unexplored_iterations","policy","exploration_stats"]

    for k in range(len(variables)):
        pickle.dump(variables[k],open('assignement_4/q_learning_results/'+signature+"_epsilon_"+str(epsilon)+"_alpha_"+str(alpha)+"_gamma_"+str(gamma)+"_mode_"+mode+"_"+names[k]+".p","wb"))

    return mdp
            

def epsilon_greedy_exploration_metrics(epsilon, state):
    Q_values_current_s = state['Q']
    best_action_indices = np.where(Q_values_current_s==Q_values_current_s.max())[0]
    if(best_action_indices.shape[0] == Q_values_current_s.shape[0]):
        chosen_action_index = rd.randint(0,best_action_indices.shape[0]-1)
        return chosen_action_index
    else:
        if np.random.uniform()<epsilon:
            action_indices = np.where(Q_values_current_s < Q_values_current_s.max())[0]
            chosen_action_index = action_indices[rd.randint(0,action_indices.shape[0]-1)]
            return chosen_action_index
        else:
            chosen_action_index = best_action_indices[rd.randint(0,best_action_indices.shape[0]-1)]
            return chosen_action_index



def performance_measure_dict(mdp,repeat=10000):
    scores = []
    for mode in ['random','intelligent'] :
        score_temp = {'win':0,'draw':0,'lost':0}
        for r in range(repeat):
            s = mdp.states_manager.get_state(0)
            while not s['terminal']:
                action_policy = s['policy']
                next_states = mdp.T(s['number'],action_policy,s['actions'],mode=mode)
                num= mdp.choose_action(next_states)
                s = mdp.states_manager.get_state(num)
            if(s['reward']==0):
                score_temp['draw']+=1
            elif(s['reward']==100):
                score_temp['win']+=1
            elif(s['reward']==-100):
                score_temp['lost'] +=1
        scores.append(score_temp)
    return scores

def update_policy(mdp):
    for key,val in mdp.states_manager.states.items():
        if(not val['terminal']):
            val['policy'] = val['actions'][np.argmax(val['Q'])]
    return

def create_policy(mdp):
    policy = {}
    for key,val in mdp.states_manager.states.items():
        if(not val['terminal']):
            policy[key] = val['actions'][np.argmax(val['Q'])]
    return policy


def exploration_statitics(mdp):
    stats = {"unexplored_actions": 0 , "mean_explored_times" : 0 ,"most_explored_times" : 0,
    "least_explored_times" : 0, "mean_std_intra" : 0}
    count=0
    for key,val in mdp.states_manager.states.items():
        if(not val['terminal']):
            count+=1
            d = val['actions_explored']
            stats['unexplored_actions']+= np.count_nonzero(d == 1)
            stats['mean_explored_times']+= np.sum(d)
            stats["most_explored_times"] = max(np.max(d),stats["most_explored_times"])
            stats["least_explored_times"] = min(np.min(d)-1,stats["least_explored_times"])
            stats['mean_std_intra'] += np.std(d)
    stats['mean_explored_times'] = stats['mean_explored_times']/count
    stats['mean_std_intra'] = stats['mean_std_intra']/count
    return stats





    












###################################################################
# WITHOUT METRICS
###################################################################
def Q_learning(mdp:Tictactoe(),step_size=0.1,epsilon =0.1,n_episodes=500,alpha=0.1):
    Q = dict([(s['number'], np.zeros(len(s['actions']))) for s in mdp.states_manager.states])
    gamma = mdp.gamma
    for episode in range(n_episodes):
        init_state = mdp.init
        state = mdp.states_manager.get_state(0)
        while not state['terminal']:
            chosen_action_index = epsilon_greedy_exploration(Q,epsilon,state)
            agent_action = state['actions'][chosen_action_index]
            potential_actions = mdp.T(state['number'],agent_action,mode='intelligent')
            final_state_number = mdp.choose_action(potential_actions)
            try:
                final_state = mdp.states_manager.get_state(final_state_number)
            except:
                print(state)
                print(chosen_action_index)
                print(agent_action)
                print(potential_actions)
                print(final_state_number)
                raise Exception("Problem")
            Q[state['number']][chosen_action_index] = Q[state['number']][chosen_action_index] + alpha*(final_state['reward']+gamma*(Q[final_state['number']].max()) - Q[state['number']][chosen_action_index])
            state = final_state
    return Q
            

def epsilon_greedy_exploration(Q, epsilon, state):
    Q_values_current_s = Q[state['number']]
    best_action_indices = np.where(Q_values_current_s==Q_values_current_s.max())[0]
    if(best_action_indices.shape[0] == Q_values_current_s.shape[0]):
        chosen_action_index = rd.randint(0,best_action_indices.shape[0]-1)
        return chosen_action_index
    else:
        if np.random.uniform()<epsilon:
            action_indices = np.where(Q_values_current_s < Q_values_current_s.max())[0]
            chosen_action_index = action_indices[rd.randint(0,action_indices.shape[0]-1)]
            return chosen_action_index
        else:
            chosen_action_index = best_action_indices[rd.randint(0,best_action_indices.shape[0]-1)]
            return chosen_action_index