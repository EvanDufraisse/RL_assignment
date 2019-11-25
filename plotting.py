import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_without_std(X,Y,labelx,labely,title,path,curves_labels=None,curves_colors=None,sizes=[(4,3)]):

    for size in sizes:
        plt.figure(figsize=size)
        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  


        if(type(X)==type([1,2,3])):
            if type(X[0]) == type(3) or type(X[0]) == type(3.2):
                single_abs= True
            else:
                single_abs= False
            

        if(type(curves_labels)==type('st')):
            curves_labels = [curves_labels]
        if(type(curves_colors)==type('st')):
            curves_colors = [curves_colors]
        if single_abs:
            for curve in range(len(Y)):
                y = Y[curve]
                x = X
                if type(curves_colors) ==type(None):
                    plt.plot( x, y, linewidth=2, label=curves_labels[curve])

                else:
                    plt.plot( x, y, color=curves_colors[curve], linewidth=2, label=curves_labels[curve])
        else:
            for curve in range(len(Y)):
                y = Y[curve]
                x = X[curve]
                if type(curves_colors) ==type(None):
                    plt.plot( x, y, linewidth=2, label=curves_labels[curve])

                else:
                    plt.plot( x, y, color=curves_colors[curve], linewidth=2, label=curves_labels[curve])


        plt.ylabel(labely, fontsize=9)
        if single_abs:
            plt.xticks(X, fontsize=9)  
        plt.title(title, fontsize=10)  
        
        plt.xlabel(labelx, fontsize=9)  
        plt.legend()
        plt.savefig(path,bbox_inches='tight')
    print('finished')
    return

