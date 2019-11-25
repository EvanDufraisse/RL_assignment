import numpy as np
from assignement_4.Tictactoe_mdp import Tictactoe
import time
import pickle

############################################################################
# WITH METRICS
############################################################################


def policy_improvement_metrics(mdp:Tictactoe(),signature,mode='random',epsilon=0.1):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    parity = 0
    U = {}
    policy = {}
    count=0
    gamma = mdp.gamma
    for key,val in mdp.states_manager.states.items():
        if not val['terminal']:
            val['policy'] = np.random.choice(val['actions'])
        val['U'+str(parity)] = 0
    last_value = parity
    new_value =abs(parity-1)
    it = 0
    iterations = []
    time_iterations = []
    policy_changes = []
    value_iterations = []
    scores_intelligent_iterations = []
    scores_random_iterations = []
    policy_change = 0
    total_time = 0
    while True:
        print("-------------------------")
        print(mode)
        print(gamma)
        print("iteration number : "+str(it))
        print("policy_change : "+str(policy_change))
        print("time_last_it : " +str(total_time))
        policy_change = 0 #metric policy change
        iterations.append(it) # metric iterations
        #-----------------------------------------------------
        start = time.time()
        print('start evaluation')
        last_value, value_it = policy_evaluation_metrics(mdp,U,last_value,epsilon)
        evaluation_time = time.time()-start
        print('evaluation time : '+str(evaluation_time))
        unchanged = True
        for key,state in mdp.states_manager.states.items():
            if(state['terminal']):
                continue
            best_action = best_action_to_make_metrics(mdp,last_value,state,mode)
            policy[key] = best_action
            if best_action != state['policy']:
                policy_change+=1
                state['policy'] = best_action
                unchanged = False
        # Iteration finished, update of metrics
        #---------------------------------------------
        total_time = time.time() - start
        time_iterations.append([evaluation_time,total_time])
        it +=1 # metric iterations
        policy_changes.append(policy_change) # metric policy change
        value_iterations.append(value_it)
        scores_temp = performance_measure_dict(policy,mdp,repeat=100000)
        scores_random_iterations.append(scores_temp[0])
        scores_intelligent_iterations.append(scores_temp[1])
        # pickle policy and value of this iteration
        name_U = "U_tic-tac-toe-eps_"+str(epsilon)+"_gamma_"+str(gamma)+"_mode_"+mode+"_iteration_"+str(it-1)
        pickle.dump(U,open("./assignement_4/value_iteration_results/"+signature+name_U+".p","wb"))
        name_P = "P_tic-tac-toe-eps_"+str(epsilon)+"_gamma_"+str(gamma)+"_mode_"+mode+"_iteration_"+str(it-1)
        pickle.dump(policy,open("./assignement_4/policy_iteration_results/"+signature+name_P+".p","wb"))        

        if unchanged:
            variables_to_pickle = [time_iterations,policy_changes,value_iterations,scores_random_iterations,
            scores_intelligent_iterations]
            names_pickle = ["time_iterations","policy_changes","value_iterations","scores_random_iterations",
            "scores_intelligent_iterations"]
            for k in range(len(names_pickle)):
                pickle.dump(variables_to_pickle[k],open("./assignement_4/policy_iteration_results/"+signature+"_"+names_pickle[k]+"eps_"+str(epsilon)+"_gamma_"+str(gamma)+"_mode_"+mode+".p","wb"))

            print("end")
            return U,policy



def policy_evaluation_metrics(mdp,U,last_value,epsilon=0.1,mode='random'):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    count=0
    new_value = abs(last_value-1)
    value_it = 0
    while True:
        delta = 0
        value_it+=1
        print(str(value_it))
        count=0
        c=1
        for key, s in mdp.states_manager.states.items():
            if(count > 1000000*c):
                c+=1
                print(count)
            count+=1
            if s['terminal']:
                s['U'+str(new_value)] =0
                continue
            action = s['policy']
            next_states = mdp.T(s['number'],s['policy'],s['actions'],mode=mode)
            value = 0
            for state in next_states:
                p = state[1]
                temp_state = mdp.states_manager.get_state(state[0])
                value += p*(temp_state['reward']+mdp.gamma*temp_state['U'+str(last_value)])
            s['U'+str(new_value)] = value
            U[key] = value
            delta = max(abs(s['U'+str(last_value)]-value),delta)
        print('delta : ' +str(delta))
        last_value = abs(last_value-1)
        new_value = abs(new_value-1)
        if delta < epsilon * (1 - gamma) / gamma:
            return last_value, value_it

def best_action_to_make_metrics(mdp,last_value,state,mode='random'):
    gamma = mdp.gamma
    s = state
    actions = s['actions']
    best_action = actions[0]
    best_value = -999
    for a in actions:
        temp_value = sum([p*(mdp.states_manager.get_state(s1)['reward'] + gamma*mdp.states_manager.get_state(s1)['U'+str(last_value)]) for (s1,p) in mdp.T(s['number'], a,s['actions'],mode=mode)])
        if temp_value > best_value:
            best_action = [a]
            best_value = temp_value
        if temp_value == best_value:
            best_action.append(a)
    #best_action = np.random.choice(np.array(best_action))
    return best_action[0]
'''
def performance_measure_dict(policy,mdp,repeat=100000):
    scores = []
    for mode in ['random','intelligent'] :
        score_temp = []
        for r in range(repeat):
            s = mdp.states_manager.get_state(0)
            while not s['terminal']:
                action_policy = policy[s['number']]
                next_states = mdp.T(s['number'],action_policy,s['actions'],mode=mode)
                num= mdp.choose_action(next_states)
                s = mdp.states_manager.get_state(num)
            score_temp.append(s['reward'])
        scores.append(score_temp)
    return scores'''
def performance_measure_dict(policy,mdp,repeat=100000):
    scores = []
    for mode in ['random','intelligent'] :
        score_temp = {'win':0,'draw':0,'lost':0}
        for r in range(repeat):
            s = mdp.states_manager.get_state(0)
            while not s['terminal']:
                action_policy = policy[s['number']]
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

############################################################################
# WITHOUT METRICS
############################################################################

def policy_improvement(mdp:Tictactoe(),mode='random',epsilon=0.00001):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    parity = 0
    U = {}
    policy = {}
    count=0
    for key,val in mdp.states_manager.states.items():
        if not val['terminal']:
            val['policy'] = np.random.choice(val['actions'])
        val['U'+str(parity)] = 0
    last_value = parity
    new_value =abs(parity-1)
    while True:
        last_value = policy_evaluation(mdp,U,last_value,epsilon)
        unchanged = True
        for key,state in mdp.states_manager.states.items():
            if(state['terminal']):
                continue
            best_action = best_action_to_make(mdp,last_value,state,mode)
            policy[key] = best_action
            if best_action != state['policy']:
                state['policy'] = best_action
                unchanged = False
        if unchanged:
            return U,policy



def policy_evaluation(mdp,U,last_value,epsilon=0.001,mode='random'):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    count=0
    new_value = abs(last_value-1)
    while True:
        delta = 0
        for key, s in mdp.states_manager.states.items():
            if s['terminal']:
                s['U'+str(new_value)] =0
                continue
            action = s['policy']
            next_states = mdp.T(s['number'],s['policy'],s['actions'],mode=mode)
            value = 0
            for state in next_states:
                p = state[1]
                temp_state = mdp.states_manager.get_state(state[0])
                value += p*(temp_state['reward']+mdp.gamma*temp_state['U'+str(last_value)])
            s['U'+str(new_value)] = value
            U[key] = value
            delta = max(abs(s['U'+str(last_value)]-value),delta)
        last_value = abs(last_value-1)
        new_value = abs(new_value-1)
        if delta < epsilon * (1 - gamma) / gamma:
            return last_value

def best_action_to_make(mdp,last_value,state,mode='random'):
    gamma = mdp.gamma
    s = state
    actions = s['actions']
    best_action = actions[0]
    best_value = -999
    for a in actions:
        temp_value = sum([p*(mdp.states_manager.get_state(s1)['reward'] + gamma*mdp.states_manager.get_state(s1)['U'+str(last_value)]) for (s1,p) in mdp.T(s['number'], a,s['actions'],mode=mode)])
        if temp_value > best_value:
            best_action = [a]
            best_value = temp_value
        if temp_value == best_value:
            best_action.append(a)
    #best_action = np.random.choice(np.array(best_action))
    return best_action[0]

