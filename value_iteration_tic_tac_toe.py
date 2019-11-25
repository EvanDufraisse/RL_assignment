import time
import random as rd
import pickle
def value_iteration(mdp, epsilon_min=0.00001,mode='intelligent',signature=""):
    "Solving an MDP by value iteration."
    # Parameters linked to metric
    epsilon = 1
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    length = len(mdp.states_manager.states)
    parity = 1
    policy = {}
    U = {}
    for key,val in mdp.states_manager.states.items():
        val['U'+str(parity)] = 0
        val['policy'] = 0
        policy[key] = 0
        U[key] = 0
    clock = []
    delta_iterations = []
    iterations = []
    it = 0
    iterations_changed_values = []
    iterations_changed_policy = []
    iterations_scores_random = []
    iterations_scores_intelligent = []
    while True:
        print(it)
        parity = abs(parity-1)
        count_changed_policy = 0
        count_changed_values = 0
        delta = 0
        start = time.time()
        count=0
        c=1
        for key,s in mdp.states_manager.states.items():
            if(count > c*1000000):
                c+=1
                print (str(count)+" over "+ str(length))
            count+=1
            if s['terminal']:
                s['U'+str(parity)] = 0
            else:
                best_value = -100
                actions_to_keep = []
                for a in s['actions']:
                    next_states = T(key,a,s['actions'],mode=mode)
                    value = 0
                    for elem in next_states:
                        temp_state = mdp.states_manager.get_state(elem[0])
                        p = elem[1]
                        value += p*(temp_state['reward']+mdp.gamma*temp_state['U'+str(abs(parity-1))])
                    if value > best_value:
                        best_value = value
                        actions_to_keep = [a]
                    elif value == best_value:
                        actions_to_keep.append(a)
                old_policy = s['policy'] 
                s['policy'] = actions_to_keep[rd.randint(0,len(actions_to_keep)-1)]
                policy[key] = s['policy']
                if(s['policy']!=old_policy):
                    count_changed_policy+=1
                s['U'+str(parity)] = best_value
                U[key] = best_value
            delta = max(delta, abs(s['U'+str(parity)] - s['U'+str(abs(parity-1))]))
            if abs(s['U'+str(parity)] - s['U'+str(abs(parity-1))]) > epsilon_min* (1 - gamma) / gamma:
                count_changed_values+=1

        it+=1
        elapsed = time.time() - start
        print(elapsed)
        clock.append(elapsed)
        
        delta_iterations.append(delta)
        iterations.append(it)
        iterations_changed_values.append(count_changed_values)
        iterations_changed_policy.append(count_changed_policy)
        scores = performance_measure_dict(policy,mdp,repeat=100000)
        iterations_scores_random.append(scores[0])
        iterations_scores_intelligent.append(scores[1])
        print (str(epsilon), end="\r")

        if delta < epsilon * (1 - gamma) / gamma:
            name_U = "U_tic-tac-toe-eps_"+str(epsilon)+"_gamma_"+str(gamma)+"_mode_"+str(mode)
            pickle.dump([delta,U],open("./assignement_4/value_iteration_results/"+signature+name_U+".p","wb"))
            name_P = "P_tic-tac-toe-eps_"+str(epsilon)+"_gamma_"+str(gamma)+"_mode_"+str(mode)
            pickle.dump([delta,policy],open("./assignement_4/value_iteration_results/"+signature+name_P+".p","wb"))

            if(epsilon == epsilon_min):
                name_iterations = "Iterations_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                name_iterations_changed_values = "Iterations_values_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                name_iterations_changed_policy = "Iterations_policy_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                name_iterations_scores_random = "Iterations_scores_random_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                name_iterations_scores_intelligent = "Iterations_scores_intelligent_tic-tac-toe_gamma_"+str(gamma)+str(mode)
                name_delta_iterations = "Iterations_delta"+str(gamma)+"_mode_"+str(mode)
                name_clock = "Iterations_clock"+str(gamma)+"_mode_"+str(mode)
                names = [name_iterations,name_iterations_changed_values,
                        name_iterations_changed_policy,
                        name_iterations_scores_random,name_iterations_scores_intelligent,name_delta_iterations,name_clock]
                elements = [iterations,iterations_changed_values,
                            iterations_changed_policy,iterations_scores_random,
                            iterations_scores_intelligent,delta_iterations,clock]
                for k in range(len(names)):
                    pickle.dump(elements[k], open("./assignement_4/value_iteration_results/"+signature+names[k]+".p", "wb"))
                return U,policy
            else:
                while(delta<epsilon * (1 - gamma) / gamma):
                    epsilon = epsilon/10
                if(epsilon<=epsilon_min):
                    name_iterations = "Iterations_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                    name_iterations_changed_values = "Iterations_values_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                    name_iterations_changed_policy = "Iterations_policy_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                    name_iterations_scores_random = "Iterations_scores_random_tic-tac-toe_gamma_"+str(gamma)+"_mode_"+str(mode)
                    name_iterations_scores_intelligent = "Iterations_scores_intelligent_tic-tac-toe_gamma_"+str(gamma)+str(mode)
                    name_delta_iterations = "Iterations_delta"+str(gamma)+"_mode_"+str(mode)
                    name_clock = "Iterations_clock"+str(gamma)+"_mode_"+str(mode)

                    names = [name_iterations,name_iterations_changed_values,
                            name_iterations_changed_policy,
                            name_iterations_scores_random,name_iterations_scores_intelligent,name_delta_iterations,name_clock]
                    elements = [iterations,iterations_changed_values,
                                iterations_changed_policy,iterations_scores_random,
                                iterations_scores_intelligent,delta_iterations,clock]
                    for k in range(len(names)):
                        pickle.dump(elements[k], open("./assignement_4/value_iteration_results/"+signature+names[k]+".p", "wb"))
                    return U,policy



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
