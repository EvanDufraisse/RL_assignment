import numpy as np
import random as rd
import MDP
def policy_improvement(mdp,epsilon=0.001):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s['number'], 0) for s in mdp.states_manager.states])
    pi = dict([(s['number'], np.random.choice(mdp.actions(s))) for s in mdp.states_manager.states])
    while True:
        U = policy_evaluation(pi, U, mdp,epsilon)
        unchanged = True
        for state_index in range(len(mdp.states_manager.states)):
            state = mdp.states_manager.states[state_index]
            if(state['terminal']):
                continue
            best_action = best_action_to_make(mdp,state_index,U)
            if best_action != pi[state['number']]:
                pi[state['number']] = best_action
                unchanged = False
        if unchanged:
            return U,pi

def policy_evaluation(pi, U, mdp,epsilon=0.001):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    while True:
        delta = 0
        U1 = U.copy()
        for state_index in range(len(mdp.states_manager.states)):
            s = mdp.states_manager.states[state_index]
            if s['terminal']:
                continue
            temp_value = U[s['number']]
            action = pi[s['number']]
            next_states = mdp.T(s['number'],action,mode='random')
            value = 0
            for state in next_states:
                p = state[1]
                state_number = state[0]
                temp_state = mdp.states_manager.get_state(state_number)
                value += p*(temp_state['reward']+mdp.gamma*U1[state_number])
            U[s['number']] = value
            delta = max(abs(temp_value-value),delta)
        if delta < epsilon * (1 - gamma) / gamma:
            return U

def best_action_to_make(mdp,state_index,U):
    gamma = mdp.gamma
    s = mdp.states_manager.states[state_index]
    actions = s['actions']
    best_action = actions[0]
    best_value = -999
    for a in actions:
        temp_value = sum([p*(mdp.states_manager.get_state(s1)['reward'] + gamma*U[s1]) for (s1,p) in mdp.T(s['number'], a,mode='random')])
        if temp_value > best_value:
            best_action = [a]
            best_value = temp_value
        if temp_value == best_value:
            best_action.append(a)
    #best_action = np.random.choice(np.array(best_action))
    return best_action[0]



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

def policy_improvement(mdp,epsilon=0.001):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s['number'], 0) for s in mdp.states_manager.states])
    pi = dict([(s['number'], np.random.choice(mdp.actions(s))) for s in mdp.states_manager.states])
    while True:
        U = policy_evaluation(pi, U, mdp,epsilon)
        unchanged = True
        for state_index in range(len(mdp.states_manager.states)):
            state = mdp.states_manager.states[state_index]
            if(state['terminal']):
                continue
            best_action = best_action_to_make(mdp,state_index,U)
            if best_action != pi[state['number']]:
                pi[state['number']] = best_action
                unchanged = False
        if unchanged:
            return U,pi

def policy_evaluation(pi, U, mdp,epsilon=0.001):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    gamma = mdp.gamma
    while True:
        delta = 0
        U1 = U.copy()
        for state_index in range(len(mdp.states_manager.states)):
            s = mdp.states_manager.states[state_index]
            if s['terminal']:
                continue
            temp_value = U[s['number']]
            action = pi[s['number']]
            next_states = mdp.T(s['number'],action,mode='random')
            value = 0
            for state in next_states:
                p = state[1]
                state_number = state[0]
                temp_state = mdp.states_manager.get_state(state_number)
                value += p*(temp_state['reward']+mdp.gamma*U1[state_number])
            U[s['number']] = value
            delta = max(abs(temp_value-value),delta)
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def policy_improvement_forest_management(mdp:MDP.ForestManagement(),epsilon=0.001):
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

def policy_evaluation_forest_management(pi, U, mdp:MDP.ForestManagement(),epsilon=0.001):
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