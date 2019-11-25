#%%







# %%
from assignement_4.pb_tic_tac_toe import state_manager
import pickle 

m = pickle.load(open('assignement_4\m.p','rb'))

# %%
import assignement_4.value_iteration as VI
from assignement_4.pb_tic_tac_toe import Tictactoe as t
init_state = '220110000'
pb = t()
actions = pb.actions(init_state)
actions_states = VI.compute_actions_states(actions,init_state,pb)
actions_results = VI.compute_actions_results(actions_states,m,[0]*m.size,0.9)



# %%
import pickle
import assignement_4.pb_tic_tac_toe as pt
from assignement_4.pb_tic_tac_toe import state_manager
m = pt.generate_states_2()
pickle.dump(m,open('assignement_4\m_tictactoe_dict_0_4_2.p','wb'))

#%%
pt.fromDeci('',3,17890)

# %%
from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    print(name)

# %%
m.get_state_rank(5)

# %%
import pickle
m = pickle.load(open('assignement_4\m_tictactoe_3.p','rb'))
for s in m.states:
    s[4]['value'] = 0
    s[4]['last_value'] = 0
import assignement_4.value_iteration as VI
VI.value_iteration(m,4)

# %%
import time
import MDP
import pickle
from assignement_4.value_iteration import value_iteration_dict
from assignement_4.policy_iteration import policy_improvement
from q_learning import Q_learning
m = pickle.load(open('assignement_4\m_tictactoe_dict_0_4_2.p','rb'))
mdp = MDP.Tictactoe(size_grid=4,states_manager=m,gamma=0.9)
#start = time.clock()

#UPI,policyPI = policy_improvement(mdp,epsilon=0.001)
#print(time.clock() - start)
#time.sleep(1)
#start = time.clock()
U,policyVI = value_iteration_dict(mdp,epsilon=0.001)
#print(time.clock() - start)
# 
# #Q = Q_learning(mdp,n_episodes=100000)


#%%
import time
import pickle
from assignement_4.Tictactoe_mdp import Tictactoe
from assignement_4.Tictactoe_mdp import *
from assignement_4.policy_iteration_tic_tac_toe import policy_improvement_metrics
#m = pickle.load(open('assignement_4\m_tictactoe_dict_0_4_2.p','rb'))
m = pickle.load(open('assignement_4\m_tictactoe_dict_reward_0_3_2.p','rb'))
print(len(m.states))
#[0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
# "t33_PI_2_"
for mode in ['random','intelligent']:
    print(mode)
    for gamma in [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]:
        print(gamma)
        mdp = Tictactoe(size_grid=3,states_manager=m,gamma=gamma)
        U,P = policy_improvement_metrics(mdp,"t33_PI_final_3",mode=mode)




# %%
import assignement_4.pb_tic_tac_toe as pt
def test_policy(string_grid,policy):
    number = pt.toDeci(string_grid,3)
    return policy[number]

# %%
import time
from assignement_4.value_iteration import value_iteration_forest_management
from assignement_4.policy_iteration import policy_improvement_forest_management
from MDP import ForestManagement
from q_learning import Q_learning_Forest_Management
mdp = ForestManagement(fire_prob=0,reward_wait=100,gamma=0.9,number_of_states=1000000)
start = time.time()
U,policyVI = value_iteration_forest_management(mdp,epsilon=0.00001)
print(time.time() - start)
time.sleep(0.1)
start = time.time()
UPI,policyPI = policy_improvement_forest_management(mdp,0.00001)
print(time.time() - start)
#Q = Q_learning_Forest_Management(mdp,n_episodes=10000,epsilon=0.1,alpha=0.5,exploration=0)
# %%
import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
l = []
for k in range(4):
    a =np.rot90(a)
    l.append(a.tobytes())
a = a.T
for i in range(4):
    a =np.rot90(a)
    l.append(a.tobytes())

print(l)


# %%
import pickle 
a = [1,2,3,4]
pickle.dump(a,open("./assignement_4/value_iteration_results/a.p","wb"))

# %%
import os
os.getcwd()

# %%
from assignement_4.value_iteration_metrics import value_iteration
import pickle
import MDP
import time
start = time.time()
for mode in ['random','intelligent']:
    for gamma in [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]:
        print(gamma)
        m = pickle.load(open("assignement_4\m_tictactoe_dict_0_3_2.p","rb"))
        mdp = MDP.Tictactoe(size_grid=3,states_manager=m,gamma=gamma)
        value_iteration(mdp,mode=mode)
print(time.time()-start)

# %%
import matplotlib.pyplot as plt
import pickle
from assignement_4.Tictactoe_mdp import Tictactoe
policy = pickle.load(open("assignement_4/value_iteration_results/P_tic-tac-toe-eps_0.01_gamma_0.9_mode_intelligent.p","rb"))
m = pickle.load(open("assignement_4\m_tictactoe_dict_0_3_2.p","rb"))
mdp = Tictactoe(size_grid=3,states_manager=m,gamma=0.9)
from assignement_4.value_iteration_metrics import performance_measure_dict
results = performance_measure_dict(policy,mdp,repeat=100000)

# %%
from assignement_4.Tictactoe_mdp import *
m = generate_states(side_size=3,print_step=1000)

# %%
#policy = pickle.load(open("assignement_4/value_iteration_results/P_tic-tac-toe-eps_0.01_gamma_0.9_mode_intelligent.p","rb"))
import matplotlib.pyplot as plt
import pickle
from assignement_4.Tictactoe_mdp import Tictactoe
m = pickle.load(open("assignement_4\m_tictactoe_dict_0_4_2.p","rb"))
mdp = Tictactoe(size_grid=4,states_manager=m,gamma=0.9)

# %%
from assignement_4.Tictactoe_mdp import *

# %%

##################################################
# COMPUTE VALUE ITERATION FOR TICTACTOE 44
##################################################
import pickle
import time
from assignement_4.Tictactoe_mdp import *
from assignement_4.value_iteration_tic_tac_toe import value_iteration
m = pickle.load(open("assignement_4\m_tictactoe_dict_0_4_2.p","rb"))
#m = pickle.load(open('assignement_4\m_tictactoe_dict_reward_0_3_2.p','rb'))
signature = 'to_throw_away'
times_value_it = {'random_0.2':2021.4412546157837}
for mode in ['random','intelligent']:
    print(mode)
    l = [0.2,0.98]
    for gamma in l :
        start = time.time()
        mdp = Tictactoe(size_grid=4,states_manager=m,gamma=gamma)
        r = value_iteration(mdp,signature=signature,mode=mode,epsilon_min=1)
        times_value_it[mode+str(gamma)] = time.time() - start
        print(times_value_it[mode+str(gamma)])





# %%
import pickle
from assignement_4.Tictactoe_mdp import *
m = pickle.load(open("assignement_4\m_tictactoe_dict_0_4_2.p","rb"))
#pickle.dump(m, open('assignement_4\m_tictactoe_dict_reward_0_3_2.p','wb'))

# %%
import pickle
import numpy as np
from assignement_4.variables.plotting import *
policy_changes = pickle.load(open('assignement_4/value_iteration_results/t44_epsilon_deterIterations_policy_tic-tac-toe_gamma_0.2_mode_random.p','rb'))
delta = pickle.load(open('assignement_4/value_iteration_results/t44_epsilon_deterIterations_delta0.2_mode_random.p',"rb"))



# %%
import pickle
#policy_changes = pickle.load(open('assignement_4/value_iteration_results/t44_epsilon_deterIterations_policy_tic-tac-toe_gamma_0.2_mode_random.p','rb'))
#delta = pickle.load(open('assignement_4/value_iteration_results/t44_epsilon_deterIterations_delta0.2_mode_random.p',"rb"))
value_changes = pickle.load(open('assignement_4/value_iteration_results/t44_epsilon_deterIterations_values_tic-tac-toe_gamma_0.2_mode_random.p','rb'))
# %%
plot_without_std(np.log(np.array(delta)),[np.array(pol)],"delta","policy changes","policy changes function of delta"
,"test_plot.png",curves_labels="p_chg",curves_colors='red')

# %%
file = 't44_epsilon_deterIterations_scores_random_tic-tac-toe_gamma_0.2_mode_intelligent.p'
scores_random =pickle.load(open('assignement_4/value_iteration_results/'+file,'rb'))
print(scores_random)
# %%

#%%

###############################################
# COMPUTE Q-LEARNING
###############################################
import time
import pickle
from assignement_4.Tictactoe_mdp import Tictactoe
from assignement_4.Tictactoe_mdp import *
from assignement_4.policy_iteration_tic_tac_toe import policy_improvement_metrics
m = pickle.load(open('assignement_4\m_tictactoe_dict_0_4_2.p','rb'))
#m = pickle.load(open('assignement_4\m_tictactoe_dict_reward_0_3_2.p','rb'))
print(len(m.states))
#l = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
l = [0.9]
step_performance_list = [0,5000,100000,1000000]
signature =  "t44_QL_"
from assignement_4.q_learning_tic_tac_toe import Q_learning_metrics
for mode in ['random','intelligent']:
    print(mode)
    for gamma in l:
        for e in [0.1]:
            print("epsilon : "+str(e))
            print("gamma : "+ str(gamma))
            mdp = Tictactoe(size_grid=4,states_manager=m,gamma=gamma)
            Q_learning_metrics(mdp,mode=mode,step_performance_list=step_performance_list,signature=signature,epsilon=e,n_episodes=1000002,alpha=0.1,big=True)




# %%
import pickle
file = 't33_QL__epsilon_0.1_alpha_0.1_gamma_0.05_score_intelligent.p'
folder = 't33_intel/'
scores_random =pickle.load(open('assignement_4/q_learning_results/'+folder+file,'rb'))
print(scores_random)

# %%
#####################################
# PLOTTINGS Q_Learning TICTACTOE
#####################################

path= "assignement_4/q_learning_results/"

#%%

################################
# Plotting VI Tictactoe
################################
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
# Plot delta function of iteration for each gamma:
#plot_without_std(X,Y,labelx,labely,title,path,curves_labels=None,curves_colors=None,sizes=[(4,3)])
# Plot delta function of iteration for each gamma:
pathVI = "assignement_4/value_iteration_results/"
#gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
gamma_t44 = [0.2]
filename = "t44_epsilon_deterIterations_policy_tic-tac-toe_gamma_"
endfile = "_mode_intelligent.p"
labels_curves=[]
Y = []
X = []
for gamma in gamma_t44:
    file_delta = filename+str(gamma)+endfile
    Y_temp = np.array(pickle.load(open(pathVI+file_delta, "rb")))
    labels_curves.append( "γ:" + str(gamma))
    Y.append(Y_temp)
    X.append(np.arange(1,Y_temp.shape[0]+1))

plot_without_std(X,Y,"Iteration","Policy Changes","Policy Changes function of iteration (Intelligent)",pathVI+'policy_it_t44_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])



# %%
# Plot scores function of iteration for each gamma:

pathVI = "assignement_4/value_iteration_results/"
#gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
gamma_t44 = [0.2]
filename = "t44_epsilon_deterIterations_scores_intelligent_tic-tac-toe_gamma_"
endfile = "intelligent.p"
labels_curves=[]
Y1 = []
X = []
Y2 = []
for gamma in gamma_t44:
    Y_temp1 = []
    Y_temp2 = []
    win_ratio = []
    win_draw_ratio = []
    file_to_pickle = filename+str(gamma)+endfile
    file_pickled = pickle.load(open(pathVI+file_to_pickle, "rb"))
    for elem in file_pickled:
        w = elem['win']
        d =  elem['draw']
        l = elem['lost']
        win_ratio.append((w+d)/(d+l+w))
        win_draw_ratio.append(l)
    Y_temp1.append(win_ratio)
    Y_temp2.append(win_draw_ratio)
    Y_temp1[0].pop(0)
    Y_temp2[0].pop(0)
    Y_temp1 = np.array(Y_temp1)
    Y_temp2 = np.array(Y_temp2)


    labels_curves.append( "γ:" + str(gamma))
    Y1.append(Y_temp1.squeeze())
    Y2.append(Y_temp2.squeeze())
    X.append(np.arange(1,Y_temp1.squeeze().shape[0]+1))

plot_without_std(X,Y1,"Iteration","Score","Win + Draw ratio function of iteration (Smart env) (Intelligent)",pathVI+'smart_win_draw_it_t44_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])
plot_without_std(X,Y2,"Iteration","Score","Number of losses function of iteration (Smart env) (Intelligent)",pathVI+'smart_losses_it_t44_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])

#%%
pathVI = "assignement_4/policy_iteration_results/"
file_to_pickle = "t33_PI_final_2_scores_random_iterationseps_0.1_gamma_0.98_mode_random.p"
endfile = "random.p"
file_pickled = pickle.load(open(pathVI+file_to_pickle, "rb"))
print(file_pickled)
# %%
################################
# Plotting PI Tictactoe
################################



#%%

##################################
# Compute VI Forest Management
##################################

from assignement_4.value_iteration_forest_management import value_iteration_forest_management_metrics
from assignement_4.Forest_management_mdp import *
for p in range(0,1,1):
    print(p)
    prob = p/1000
    for gamma in [0.9999]:
        print(gamma)
        for n in [100000]:
            mdp = ForestManagement(number_of_states=n,reward_cut=2,reward_wait=100,fire_prob=prob,gamma=gamma)
            value_iteration_forest_management_metrics(mdp,'to_throw',epsilon=0.0000001)

# %%
import pickle
pathVIFM = "assignement_4/policy_iteration_results/"
file_to_pickle = "t44_PI__time_iterationseps_0.1_gamma_0.9_mode_intelligent.p"
file_pickled = pickle.load(open(pathVIFM+file_to_pickle, "rb"))
print(file_pickled[-1])
somme = 0
for k in range(len(file_pickled)):
    somme+=file_pickled[k][-1]
print(somme)

# %%
################################
# Plotting VI FM
################################
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
pathVIFM = "assignement_4/forest_management/value_iteration/"
#gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]0.1,0.2,0.4,0.5,0.7,0.8,
gamma_FM = [0.9,0.99,0.999,0.9999]
#filename = "FM_VIdelta_iterations_gamma_"+p+"_fire_prob_"+gamma+".p"
for p in range(0,13,1):
    labels_curves=[]
    Y = []
    X = []
    for gamma in [0.1,0.2,0.4,0.6,0.8,0.9,0.9999]:
        file_delta = "FM_VInumber_policy_wait_iterations_gamma_"+str(gamma)+"_fire_prob_"+str(p/1000)+".p"
        Y_temp = pickle.load(open(pathVIFM+file_delta, "rb"))
        labels_curves.append( "wait")
        Y.append(Y_temp[-1])
        X.append(gamma)
    Y = [np.array(Y)]
    plot_without_std(X,Y,"γ value","Number state converged to wait","Number state converged Changes function of γ for p_fire="+str(p/1000),pathVIFM+'state_conv_it_FM'+str(p)+'.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])

# %%

##################################
# Plotting PI TTT
##################################
import numpy as np
import pickle
pathPITTT = "assignement_4/policy_iteration_results/"

gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
gamma_t44 = [0.2,0.5,0.9]
#file_name = 't33_PI_2__scores_intelligent_iterationseps_0.1_gamma_'+gamma+'_mode_'+mode+'.p'
results_random = {}

results_intelligent = {}
results= {'random':results_random, 'intelligent':results_intelligent}
for gamma in gamma_t44:
    print(gamma)
    for mode in ['random','intelligent']:
        liste = []
        dico = results[mode]
        file_name = 't44_PI__scores_intelligent_iterationseps_0.1_gamma_'+str(gamma)+'_mode_'+mode+'.p'
        array = pickle.load(open(pathPITTT+file_name, 'rb'))
        for k in range(len(array)):
            a_temp = np.array(array[k]).squeeze()
            temp_dic = {}
            temp_dic['win'] = np.count_nonzero(a_temp == 100)
            temp_dic['draw'] = np.count_nonzero(a_temp == 0)
            temp_dic['lost'] = np.count_nonzero(a_temp ==-100)
            liste.append(temp_dic)
        dico[gamma] = liste.copy()
#%%
from assignement_4.plotting import plot_without_std
filename = "t44_epsilon_deterIterations_scores_intelligent_tic-tac-toe_gamma_"
endfile = "intelligent.p"
labels_curves=[]
Y1 = []
X = []
Y2 = []
mode = 'intelligent'
for gamma in gamma_t44:
    Y_temp1 = []
    Y_temp2 = []
    win_ratio = []
    win_draw_ratio = []
    file_pickled = results[mode][gamma]
    for elem in file_pickled:
        w = elem['win']
        d =  elem['draw']
        l = elem['lost']
        win_ratio.append((w+d)/(d+l+w))
        win_draw_ratio.append((w+d)/(d+l+w))
    Y_temp1.append(win_ratio)
    Y_temp2.append(win_draw_ratio)
    Y_temp1[0].pop(0)
    Y_temp2[0].pop(0)
    Y_temp1 = np.array(Y_temp1)
    Y_temp2 = np.array(Y_temp2)


    labels_curves.append( "γ:" + str(gamma))
    Y1.append(Y_temp1.squeeze())
    Y2.append(Y_temp2.squeeze())
    X.append(np.arange(1,Y_temp1.squeeze().shape[0]+1))

plot_without_std(X,Y1,"Iteration","Score","Win + Draw ratio function of iteration (Smart player2) (Intelligent)",pathPITTT+'random_win_it_t44_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])
plot_without_std(X,Y2,"Iteration","Score","Number of losses function of iteration (Smart player2) (random env)",pathPITTT+'smart_losses_it_t44_random.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])

            


#%%
##################################
# Plotting PI TTT 2
##################################
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
# Plot delta function of iteration for each gamma:
#plot_without_std(X,Y,labelx,labely,title,path,curves_labels=None,curves_colors=None,sizes=[(4,3)])
# Plot delta function of iteration for each gamma:
pathVI = "assignement_4/value_iteration_results/"
gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
#gamma_t44 = [0.2]
filename = "t44_epsilon_deterIterations_policy_tic-tac-toe_gamma_"
endfile = "_mode_intelligent.p"
labels_curves=[]
Y = []
X = []
for gamma in gamma_t33:
    file_delta = filename+str(gamma)+endfile
    Y_temp = np.array(pickle.load(open(pathVI+file_delta, "rb")))
    labels_curves.append( "γ:" + str(gamma))
    Y.append(Y_temp)
    X.append(np.arange(1,Y_temp.shape[0]+1))

plot_without_std(X,Y,"Iteration","Policy Changes","Policy Changes function of iteration (Intelligent)",pathVI+'policy_it_t44_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])

#%%

# %%
# Plot scores function of iteration for each gamma:
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
pathVIFM = "assignement_4/policy_iteration_results/"
gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
#gamma_t44 = [0.2]
filename = "t33_PI_final_3_scores_random_iterationseps_0.1_gamma_"
endfile = "_mode_intelligent.p"
labels_curves=[]
Y1 = []
X = []
Y2 = []
for gamma in gamma_t33:
    Y_temp1 = []
    Y_temp2 = []
    win_ratio = []
    win_draw_ratio = []
    file_to_pickle = filename+str(gamma)+endfile
    file_pickled = pickle.load(open(pathVIFM+file_to_pickle, "rb"))
    for elem in file_pickled:
        w = elem['win']
        d =  elem['draw']
        l = elem['lost']
        win_ratio.append((w)/(d+l+w))
        win_draw_ratio.append(l)
    Y_temp1.append(win_ratio)
    Y_temp2.append(win_draw_ratio)
    Y_temp1[0].pop(0)
    Y_temp2[0].pop(0)
    Y_temp1 = np.array(Y_temp1)
    Y_temp2 = np.array(Y_temp2)


    labels_curves.append( "γ:" + str(gamma))
    Y1.append(Y_temp1.squeeze())
    Y2.append(Y_temp2.squeeze())
    X.append(np.arange(1,Y_temp1.squeeze().shape[0]+1))

plot_without_std(X,Y1,"Iteration","Score","Win ratio function of iteration (Random player2) (Intelligent)",pathVIFM+'random_win_draw_it_t33_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])
#plot_without_std(X,Y2,"Iteration","Score","Number of losses function of iteration (Smart env) (Intelligent)",pathVIFM+'smart_losses_it_t33_intel.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])

# %%
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
# Plot delta function of iteration for each gamma:
#plot_without_std(X,Y,labelx,labely,title,path,curves_labels=None,curves_colors=None,sizes=[(4,3)])
# Plot delta function of iteration for each gamma:
pathVIFM = "assignement_4/policy_iteration_results/"
gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]
gamma_t44 = [0.2]
filename = "t33_PI_final_2_policy_changeseps_0.1_gamma_"
endfile = "_mode_random.p"
labels_curves=[]
Y = []
X = []
for gamma in gamma_t33:
    file_delta = filename+str(gamma)+endfile
    Y_temp = np.array(pickle.load(open(pathVIFM+file_delta, "rb")))
    labels_curves.append( "γ:" + str(gamma))
    Y.append(Y_temp)
    X.append(np.arange(1,Y_temp.shape[0]+1))

plot_without_std(X,Y,"Iteration","Policy Changes","Policy Changes function of iteration (Random)",pathVIFM+'policy_it_t33_random.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])


# %%
##################################
# Compute PI Forest Management
##################################

from assignement_4.value_iteration_forest_management import value_iteration_forest_management_metrics
from assignement_4.policy_iteration_forest_management import policy_improvement_forest_management_metrics
from assignement_4.Forest_management_mdp import * #0.1,0.2,0.4,0.6,0.8,0.9,
for p in range(0,1,1):
    print(p)
    prob = p/1000
    for gamma in [0.9999]:
        print(gamma)
        for n in [300,10000,100000]:
            mdp = ForestManagement(number_of_states=n,reward_cut=2,reward_wait=100,fire_prob=prob,gamma=gamma)
            policy_improvement_forest_management_metrics(mdp,'fuezgfshi',epsilon=0.0000001)


# %%
# %%
################################
# Plotting PI FM
################################
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
pathPIFM = "assignement_4/forest_management/"
#gamma_t33 = [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]0.1,0.2,0.4,0.5,0.7,0.8,
#gamma_FM = [0.9,0.99,0.999,0.9999]
#filename = "FM_VIdelta_iterations_gamma_"+p+"_fire_prob_"+gamma+".p"
for p in range(0,13,1):
    labels_curves=[]
    Y = []
    X = []
    for gamma in [0.1,0.2,0.4,0.6,0.8,0.9,0.99]:
        file_delta = "PI_FM_1_number_policy_wait_iterations_gamma_"+str(gamma)+"_fire_prob_"+str(p/1000)+".p"
        Y_temp = pickle.load(open(pathPIFM+file_delta, "rb"))
        labels_curves.append( "wait")
        Y.append(Y_temp[-1])
        X.append(gamma)
    Y = [np.array(Y)]
    plot_without_std(X,Y,"γ value","Number state converged to wait","Number state converged Changes function of γ for p_fire="+str(p/1000),pathPIFM+'state_conv_it_FM'+str(p)+'.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])


# %%
        
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
pathQ = "assignement_4/q_learning_results/t33_rand/"
for epsilon in [0.1,0.5,0.8]:
    labels_curves=[]
    Y = []
    X = []
    for gamma in [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]:
        win_ratio = []
        step_iteration_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_step_iteration.p"
        time_step_iteration_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_time_step_iteration.p"
        score_random_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_score_random.p"
        step_iteration = pickle.load(open(pathQ+step_iteration_filename, "rb"))
        time_step_iteration = pickle.load(open(pathQ+time_step_iteration_filename, "rb"))
        score_random = pickle.load(open(pathQ+score_random_filename,"rb"))
        for elem in score_random:
            w = elem['win']
            d =  elem['draw']
            l = elem['lost']
            win_ratio.append((w+d)/(d+l+w))
        labels_curves.append( "γ:" + str(gamma))
        Y.append(np.array(win_ratio).squeeze())
        X.append(np.array(time_step_iteration).squeeze())
    plot_without_std(X,Y,"Time","Win+Draw score","Ratio Win+Draw (Random player2) ε ="+str(epsilon)+"(Random)",pathQ+'Ql_time_score_rand_t33_'+str(epsilon)+'random.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])


# %%
import pickle
import numpy as np
from assignement_4.plotting import plot_without_std
pathQ = "assignement_4/q_learning_results/t33_rand/"
for gamma in [0.01,0.05,0.1,0.4,0.5,0.7,0.9,0.98]:
    labels_curves=[]
    Y = []
    X = []
    for epsilon in [0.1,0.5,0.8]:
        win_ratio = []
        step_iteration_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_step_iteration.p"
        time_step_iteration_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_time_step_iteration.p"
        score_random_filename = "t33_QL__epsilon_"+str(epsilon)+"_alpha_0.1_gamma_"+str(gamma)+"_unexplored_iterations.p"
        step_iteration = pickle.load(open(pathQ+step_iteration_filename, "rb"))
        time_step_iteration = pickle.load(open(pathQ+time_step_iteration_filename, "rb"))
        score_random = pickle.load(open(pathQ+score_random_filename,"rb"))
        labels_curves.append( "ε:" + str(epsilon))
        Y.append(np.array(score_random).squeeze())
        X.append(np.arange(0,len(score_random)))
    plot_without_std(X,Y,"Iteration","Unexplored states","Unexplored states γ ="+str(gamma),pathQ+'Ql_it_score_rand_t33_'+str(gamma)+'unexplored.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])


# %%
##################################
# Compute QL Forest Management
##################################

from assignement_4.q_learning import Q_learning_Forest_Management
from assignement_4.Forest_management_mdp import *
l = {}
for p in range(0,13,1):
    print(p)
    temp_list = []
    prob = p/1000
    for gamma in [0.1,0.2,0.4,0.6,0.8,0.9,0.99]:
        mdp = ForestManagement(number_of_states=300,reward_cut=2,reward_wait=100,fire_prob=prob,gamma=gamma)
        temp_list.append(Q_learning_Forest_Management(mdp,n_episodes=5000,exploration=1))
    l[p] = temp_list

# %%

import numpy as np
from assignement_4.plotting import plot_without_std
pathQ = "assignement_4/q_learning_results/t33_rand/"
labels_curves =[]
X = []
Y = []
gammas = [0.1,0.2,0.4,0.6,0.8,0.9,0.99]
for k in range(len(gammas)) :
    Y=[]
    X =[]
    labels_curves =[]
    gamma = gammas[k]
    labels_curves.append("γ: "+str(gamma))
    Y.append(np.array(l[k]).squeeze())
    X.append(np.arange(1,len(l[k])+1))
    plot_without_std(X,Y,"Iteration","Policy Changes","Policy Changes function of iteration (Random)",pathQ+'policy_it_t33_random.png',curves_labels=labels_curves,curves_colors=None,sizes=[(4,3)])


# %%
