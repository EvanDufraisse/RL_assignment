########################################################################
# IMPORTS
########################################################################
import numpy as np



########################################################################
# PROBLEM GENERATION PARAMETERS
########################################################################
SIZE = 3
NUMBER_SQUARES = SIZE**2
NUMBER_STATES = 3**(NUMBER_SQUARES)
size_board = NUMBER_SQUARES
side_size =SIZE

'''Know that X=1 and O=2'''
from enum import Enum
class state_status(Enum):
	UNFINISHED = 0
	X = 1
	O = 2

########################################################################
# DECIMAL TO TERNARY AND VICE VERSA
########################################################################
def val(c): 
    if c >= '0' and c <= '9': 
        return ord(c) - ord('0') 
    else: 
        return ord(c) - ord('A') + 10;
    
def toDeci(str,base): 
    llen = len(str) 
    power = 1 #Initialize power of base 
    num = 0     #Initialize result 
  
    # Decimal equivalent is str[len-1]*1 +  
    # str[len-1]*base + str[len-1]*(base^2) + ...  
    for i in range(llen - 1, -1, -1): 
          
        # A digit in input number must  
        # be less than number's base  
        if val(str[i]) >= base: 
            print('Invalid Number') 
            return -1
        num += val(str[i]) * power 
        power = power * base 
    return num 

def reVal(num): 
  
    if (num >= 0 and num <= 9): 
        return chr(num + ord('0')); 
    else: 
        return chr(num - 10 + ord('A')); 
  
# Utility function to reverse a string 
def strev(str): 
  
    len = len(str); 
    for i in range(int(len / 2)): 
        temp = str[i]; 
        str[i] = str[len - i - 1]; 
        str[len - i - 1] = temp; 
  
# Function to convert a given decimal  
# number to a base 'base' and 
def fromDeci(res, base, inputNum): 
  
    index = 0; # Initialize index of result 
  
    # Convert input number is given base  
    # by repeatedly dividing it by base  
    # and taking remainder 
    while (inputNum > 0): 
        res+= reVal(inputNum % base); 
        inputNum = int(inputNum / base); 
  
    # Reverse the result 
    res = res[::-1]; 
  
    return res; 
########################################################################
# GENERATION
########################################################################
'''
Check if a given state is valid, namely, same number of X and O
Is used for generation
'''
def is_valid(state_string,size_board):
    state = state_string
    count_X = 0
    count_O = 0
    for i in range(len(state)):
        if state[i] == '1':
            count_X += 1
        elif state[i] == '2':
            count_O += 1
    return [count_X == count_O, size_board -count_X-count_O]


'''
Add zeros to form grid of suitable size
'''
def add_zeros_to_state(state_string,size_board):
    state = state_string
    to_add = size_board - len(state)
    state = '0'*to_add + state
    return state

def to_grid(state_string,side_size,size_board):
    state = add_zeros_to_state(state_string,size_board)
    grid = []
    temp = []
    for i in range(len(state)):
        if(i%side_size != side_size-1):
            temp.append(state[i])
        else:
            temp.append(state[i])
            grid.append(temp)
            temp = []
    return grid

def winner(grid,side_size):
    victory_X = 0
    victory_O = 0
    # Check rows and columns
    for i in range(side_size):
        row_val = grid[i][0]
        col_val = grid[0][i]
        row_win = True
        col_win = True
        for j in range(side_size):
            row_win = row_win and row_val == grid[i][j]
            col_win = col_win and col_val == grid[j][i]
        
        if row_win:
            if row_val == '1':
                victory_X +=1
            elif row_val == '2':
                victory_O +=1
        if col_win:
            if col_val == '1':
                victory_X+=1
            elif col_val == '2':
                victory_O+=1
    if victory_O >= 1:
        return state_status.O
    elif victory_X >=1:
        return state_status.X
    else:
        # Check diagonals
        diag1_val = grid[0][0]
        diag2_val = grid[side_size-1][0]
        diag1_win = True
        diag2_win = True
        for i in range(side_size):
            diag1_win = diag1_win and diag1_val == grid[i][i]
            diag2_win = diag2_win and diag2_val == grid[side_size-1-i][i]
        if diag1_win:
            if diag1_val == '1':
                victory_X +=1
            elif diag1_val == '2':
                victory_O +=1
        if diag2_win:
            if diag2_val == '1':
                victory_X+=1
            elif diag2_val == '2':
                victory_O+=1
        if victory_O >= 1:
            return state_status.O
        elif victory_X >=1:
            return state_status.X
        else:
            return state_status.UNFINISHED

def actions(state_string):
    state = state_string
    indices = []
    for i in range(len(state)):
        if state[i] == '0':
            indices.append(i)
    return indices

def next_states(state_string,index,indices):
        next_states_list = []
        state = state_string
        for i in indices:
            state_list = list(state)
            state_list[index] = '2'
            state_list[i] = '1'
            next_states_list.append(''.join(state_list))
        return next_states_list

def print_grid(state,side_size):
    state_to_print = ''
    temp = ''
    for i in range(len(state)):
        if(i%side_size != side_size-1):
            if state[i] == '1':
                temp+=" X    " 
            elif state[i] == '2':
                temp+= " O    "
            else:
                temp+="("+str(i)+")   " 
        else:
            if state[i] == '1':
                temp+=" X " 
            elif state[i] == '2':
                temp+= " O "
            else:
                temp+="("+str(i)+")"
            state_to_print = state_to_print+temp+"\n"
            temp = ''
    return state_to_print

class state_manager_dict:
    
    def __init__(self):
        self.states = {}
        self.size = 0
        return
    
    def add_state(self,state):
        if(state['number'] in self.states):
            pass
        else:
            self.states[state['number']] = state
            self.size+=1

    def get_state(self,num):
        return self.states[num]

def generate_states(side_size=4,print_step = 100000):
    size_board = side_size**2
    tot_combinatorial_number = 3**(side_size**2)
    manager = state_manager_dict()
    count=0
    for k in range(tot_combinatorial_number):
        count+=1
        if(count == print_step):
            print(k)
            count=0
        state = fromDeci('',3,k)
        grid_string = add_zeros_to_state(state,size_board)
        [valid,empty_cells] = is_valid(grid_string,size_board)
        if(valid):
            if(empty_cells>1):
                win = winner(to_grid(grid_string,side_size,size_board),side_size)
                if(win.value == state_status.O.value):
                    manager.add_state({'number':k,'terminal':True,'reward':100,'actions':actions(grid_string)})
                elif(win.value == state_status.X.value):
                    manager.add_state({'number':k,'terminal':True,'reward':-100,'actions':actions(grid_string)})
                else:
                    manager.add_state({'number':k,'terminal':False,'reward':0,'actions':actions(grid_string)})
            else:
                state_check_draw = ''
                for i in range(len(grid_string)):
                    if grid_string[i] == '0':
                        state_check_draw = state_check_draw + '2'
                    state_check_draw = state_check_draw + grid_string[i]
                win = winner(to_grid(grid_string,side_size,size_board),side_size)
                if(win.value == state_status.UNFINISHED.value):
                    manager.add_state({'number':k,'terminal':True,'reward':0,'actions':actions(grid_string)})
                else:
                    manager.add_state({'number':k,'terminal':True,'reward':100,'actions':actions(grid_string)})
    return manager


########################################################################
# MDP
########################################################################

class Tictactoe():
    def __init__(self,size_grid=3,states_manager=None,gamma=0.9):
        self.init = '0'*(size_grid**2)
        self.states_manager = states_manager
        self.gamma = gamma
        self.size_grid = size_grid
        self.index_state_highest = self.size_grid**2-1
    
    def R(self,state):
        return state['reward']

    def T(self, state_number, action,state_actions,mode='intelligent',):
            temp_state_number = state_number + 2*3**(self.index_state_highest-action)
            actions = state_actions.copy()
            actions.remove(action)
            if mode != 'intelligent':
                # The opponent is supposed to play randomly
                p = 1/len(actions)
                list_actions_prob = []
                for a in actions:
                    list_actions_prob.append((temp_state_number+1*3**(self.index_state_highest-a),p))
                return list_actions_prob
            else:
                #See if you can win
                for a in actions:
                    temp_next_state = self.states_manager.get_state(temp_state_number+1*3**(self.index_state_highest-a))
                    reward_next_state = temp_next_state['reward']
                    if(reward_next_state == -100):
                        return [(temp_next_state['number'],1)]

                # Otherwise check if player1 wins at some places that you should block
                if(len(actions)>2):
                    if(len(actions)==4):
                        l=set()
                        for k in range(len(actions)):
                            temp_next_state = self.states_manager.get_state(temp_state_number+2*3**(self.index_state_highest-actions[k])+
                            3**(self.index_state_highest-actions[(k+1)%(len(actions))]) +
                            3**(self.index_state_highest-actions[(k+2)%(len(actions))]))
                            reward_next_state = temp_next_state['reward']
                            if(reward_next_state == 100):
                                l.add((k+1)%(len(actions)))
                                l.add((k+2)%(len(actions)))
                        if(len(l)==4):
                            pass
                        else:
                            for k in range(len(actions)):
                                if not (k in l):
                                    return [(temp_state_number+3**(self.index_state_highest-actions[k]),1)]
                    else:
                        for k in range(len(actions)):
                            temp_next_state = self.states_manager.get_state(temp_state_number+2*3**(self.index_state_highest-actions[k])+
                            3**(self.index_state_highest-actions[(k+1)%(len(actions))]) +
                            3**(self.index_state_highest-actions[(k+2)%(len(actions))]))
                            reward_next_state = temp_next_state['reward']
                            if(reward_next_state == 100):
                                return [(temp_state_number+3**(self.index_state_highest-actions[k]),1)]
                elif(len(actions)>1):
                    for k in range(len(actions)):
                        temp_next_state = self.states_manager.get_state(temp_state_number+
                        3**(self.index_state_highest-actions[(k+1)%(len(actions))]))
                        reward_next_state = temp_next_state['reward']
                        if(reward_next_state == 100):
                            return [(temp_state_number+3**(self.index_state_highest-actions[k]),1)]

                # Finally if nothing interesting to play considering one round ahead, play randomly
                p = 1/len(actions)
                list_actions_prob = []
                for a in actions:
                    list_actions_prob.append((temp_state_number+3**(self.index_state_highest-a),p))
                return list_actions_prob

    def actions(self,state):
        return state['actions']
        
    def next_state_after_action(self,state,action,player=2):
        return state['number'] + player*3**(self.index_state_highest-action)

    def choose_action(self,T_output):
        a = np.random.uniform()
        sum_prob = 0
        for elem in T_output:
            sum_prob += elem[1]
            if(sum_prob>=a):
                return elem[0]
        return elem[0]


def play_against(policy,mdp:Tictactoe()):
    s = mdp.states_manager.get_state(0)
    init = True
    while s['terminal'] != True:
        action = policy[s['number']]
        if init:
            state_string = '000000000'
            init = False
        else:
            state_string = add_zeros_to_state(fromDeci('',3,s['number']),mdp.size_grid**2)
        state_list = list(state_string)
        state_list[action] = '2'
        state_string = ''.join(state_list)
        print(print_grid(state_string,mdp.size_grid))
        case = input("action ?")
        state_list = list(state_string)
        state_list[int(case)] = '1'
        state_string = ''.join(state_list)
        s = mdp.states_manager.get_state(toDeci(state_string,3))
    return s['reward']


def play_against_intel(mdp:Tictactoe()):
    s = mdp.states_manager.get_state(0)
    init = True
    while s['terminal'] != True:
        if init:
            state_string = '0'*mdp.size_grid**2
            init = False
        else:
            state_string = add_zeros_to_state(fromDeci('',3,s['number']),mdp.size_grid**2)
        print(print_grid(state_string,mdp.size_grid))
        case = input("action ?")
        next_states = mdp.T(s['number'],int(case),s['actions'])
        agent_action = mdp.choose_action(next_states)
        s = mdp.states_manager.get_state(agent_action)
    return s['reward']