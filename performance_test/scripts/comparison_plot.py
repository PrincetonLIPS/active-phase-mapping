import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from copy import copy
from glob import iglob

def plot_comparison(master_df):
    width=10
    height=6
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(width, height))
    #plt.title(title)
    ax = axs[0]
    ax=sns.lineplot(
        data=master_df, x="index", y='energy_error_means', palette=hue_palette, hue="Label", legend=False, errorbar="sd",linewidth=linewidth,ax=axs[0])
    ax.tick_params(direction='in',which='both',labelsize=colors['size'],size=6,width=0.5)
    ax.set_xlim(0,200)
    ax.set_ylim(-0.01,0.5)
    ax.set_aspect(1.0/ax.get_data_ratio())
    ax.set_xlabel('Iteration',size=colors['size'])
    ax.set_ylabel('Mean Average Error',size=colors['size'])
    ax.set_xticks([0,50,100,150,200])


    ax = axs[1]
    ax=sns.lineplot(
        data=master_df, x="index", y='true_positive_rate', palette=hue_palette, hue="Label", legend=False, errorbar="sd",linewidth=linewidth,ax=axs[1])
    ax.tick_params(direction='in',which='both',labelsize=colors['size'],size=6,width=0.5)
    ax.set_xlim(0,200)

    ax.set_aspect(1.0/ax.get_data_ratio())
    ax.set_xlabel('Iteration',size=colors['size'])
    ax.set_ylabel('True Positive Rate',size=colors['size'])
    ax.set_xticks([0,50,100,150,200])
    ax.set_yticks([0.2,0.4,0.6,0.8,1])
    #ax.set_ylim(0,1)

    ax = axs[2]
    ax=sns.lineplot(
        data=master_df, x="index", y='false_positive_rate', palette=hue_palette, hue="Label", legend=False, errorbar="sd",linewidth=linewidth,ax=axs[2])
    ax.tick_params(direction='in',which='both',labelsize=colors['size'],size=6,width=0.5)
    #ax.set_ylim(0,1)
    ax.set_xlim(0,200)
    ax.set_ylim(-0.01,0.2)
    ax.set_aspect(1.0/ax.get_data_ratio())
    ax.set_ylabel('False Positive Rate',size=colors['size'])
    ax.set_xlabel('Iteration',size=colors['size'])
    ax.set_xticks([0,50,100,150,200])



    plt.subplots_adjust(left=0.125, bottom=0.075, right=0.9, top=0.987, wspace=0.338, hspace=0.13)

    plt.savefig('../figs/2D_3P.png',dpi=300)
    plt.show()



#master_df=master_df.reset_index()

colors={'size':12,'xlim':(0.0,1.0),'num_stds':2,'aspect':1,'markers':['o','s','d'],'hull':'grey','energies':[[12/255, 7/255, 134/255],[182/255, 47/255, 139/255],[251/255, 163/255, 55/255]],'background':[245/255, 245/255, 245/255], 'tangent':'black', 'mu_A':'tab:red', 'mu_B':'tab:orange'}
hue_palette = {'BASE': colors['energies'][1], 'ALCH': colors['energies'][0], 'FPS':colors['energies'][2]}
linewidth=2.
###Plotting
rcParams['font.family'] = 'Arial'


dfs=[]
path_lst=[i for i in iglob('../data/*/*.csv')]
for index,path in enumerate(path_lst):
    df=pd.read_csv(path).reset_index()
    if path.split('/')[-2]=='base':
        df['Label']='BASE'
    if path.split('/')[-2]=='fps':
        df['Label']='FPS'
    elif path.split('/')[-2]=='cal':
        df['Label']='ALCH'
    df=df[:-2]#removing last two rows--often incomplete
    title=df['Seed'][0]
    #plot_comparison(df)
    dfs.append(df)
master_df=pd.concat(dfs)
#for i in [12,22,45,50,35]:
#    master_df=master_df[master_df['Seed']!=i]
plot_comparison(master_df)
