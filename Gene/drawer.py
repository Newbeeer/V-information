import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='darkgrid', font_scale=1.5,rc={"lines.linewidth": 2.5})
F = np.load('F_20.npy')[:9]

#F = np.load('F_30.npy')
KSG = np.load('K_30.npy')
mutual_kde = np.load('KDE_20.npy')[:9]
partitioning = np.load('P_30.npy')
Mixed_KSG = np.load('MK_30.npy')
Noise_KSG = np.load('NK_30.npy')
#JVHW = np.load('JVHW_discrete.npy')
#MLE = np.load('3HMLE_discrete.npy')
plt.figure(figsize=(16,16))
s = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 60,
}
wid=3
size = 20
plt.plot(s,F,label = '$\mathcal{V}-information$',marker='D',linewidth=6,linestyle='--')
plt.plot(s,KSG,label = 'KSG',marker='o',linewidth=wid)
plt.plot(s,mutual_kde,label = 'KDE',marker='v',linewidth=wid)
plt.plot(s,partitioning,label = 'Partitioning',marker='x',linewidth=wid)
plt.plot(s,Mixed_KSG,label = 'Mixed_KSG',marker='s',linewidth=wid)
#plt.plot(s,mlp,label = 'MLP',marker='s')
#plt.plot(s,Noise_KSG,label = 'Noisy_KSG',marker='^')
#plt.plot(s,JVHW,label = 'JVHW')
#plt.plot(s,MLE,label = 'MLE')
plt.tick_params(labelsize=35)

plt.xlabel('Sample Size',font1)
plt.ylabel('AUC',font1)
plt.legend(fontsize=50)
plt.savefig('gene.pdf',dpi=300, bbox_inches=None)

