import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from copy import copy


lst=['../data/chaase_meanfield_5_5.csv']
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
plt.plot(master_df['iteration'],master_df['means'],linewidth=2.25)
plt.plot(master_df['iteration'],master_df['RMSE'],linewidth=2.25)
plt.xlabel('Iteration',size=size)
plt.ylabel('Mean Error',size=size)
plt.tick_params(direction='in',which='both',labelsize = size)
#ax.set_aspect(1.0/ax.get_data_ratio())
plt.savefig('../figs/error.png',dpi=600)
plt.show()
