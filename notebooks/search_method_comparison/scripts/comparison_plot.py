import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from copy import copy


lst=['../data/base.csv','../data/random.csv','../data/fps.csv']#, '../data/chaase_meanfield_5_5.csv', 'chaase_joint_entropy_2_2.csv']
df_lst=[]
for x in lst:
    type=(x.split('/')[-1]).split('.')[0]
    tmp=pd.read_csv(x)
    tmp['Type']=type
    df_lst.append(tmp)
master_df=pd.concat(df_lst)

###Plotting
size=14
rcParams['font.family'] = 'Arial'

#entropy
plt.figure()
ax=sns.lineplot(
   data=master_df, x="iteration", y="Initial_Entropy", hue="Type", palette='deep', legend=True,linewidth=2.25)
plt.xlabel('Iteration',size=size)
plt.ylabel('Entropy',size=size)
plt.tick_params(direction='in',which='both',labelsize = size)
ax.set_aspect(1.0/ax.get_data_ratio())
plt.savefig('../figs/entropy_comparison.png',dpi=600)
#plt.show()

#error
plt.figure()
ax=sns.lineplot(
   data=master_df, x="iteration", y="means", hue="Type", palette='deep', legend=True,linewidth=2.25)
plt.xlabel('Iteration',size=size)
plt.ylabel('Mean Error',size=size)
plt.tick_params(direction='in',which='both',labelsize = size)
ax.set_aspect(1.0/ax.get_data_ratio())
plt.savefig('../figs/error_comparison.png',dpi=600)
plt.show()
