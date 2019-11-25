#################################################
# IMPORTS
#################################################
import numpy as np


#################################################
# MDP
#################################################


class ForestManagement():
    def __init__(self,states_manager=None,number_of_states=15,reward_cut=2,reward_wait=4,fire_prob=0.1,gamma=0.9):
        self.init = 0
        self.states_manager = states_manager
        self.gamma = gamma
        self.number_of_states = number_of_states
        self.reward_wait = reward_wait
        self.reward_cut = reward_cut
        self.fire_prob = fire_prob

    
    def R(self,state_number,action):
        if action == 'cut':
            return self.reward_cut
        else:
            return 0
    

    def T(self, state_number, action):
        if action == 'cut' and state_number != self.number_of_states:
            return 
           

    def actions(self,state_index):
        if type(state_index) == int:
            return self.states_manager.states[state_index]['actions']
        else:
            return state_index['actions']
        
    def next_state_after_action(self,state,action,player=2):
        return state['number'] + player*3**(self.index_state_highest-action)

    def choose_action(self,T_output):
        a = np.random.uniform()
        sum_prob = 0
        for elem in T_output:
            sum_prob += elem[1]
            if(sum_prob<=a):
                return elem[0]
        return elem[0]

    def get_state(self,state_number):
            return {'number':state_number,'terminal':(state_number == self.number_of_states-1)}
