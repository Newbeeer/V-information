import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='darkgrid', font_scale=1.5,rc={"lines.linewidth": 2.5})

def fine(a,idx,cnt,b):

    a_ = np.zeros((20))
    for i in range(1,20):

        if idx + i < 20:

            a_[i] += (b[idx+i] - a[idx+i]) # H(Y) - H(Y|X)
            cnt[i] += 1
        if idx - i >= 0:
            a_[i] += (b[idx-i] - a[idx-i]) # H(Y) - H(Y|X)
            cnt[i] += 1

    return a_,cnt

cnt = np.zeros(20)

b = np.load('frames_H_2.npy')
b += np.load('frames_H_3.npy')
b += np.load('frames_H_4.npy')
b += np.load('frames_H_5.npy')
b += np.load('frames_H_6.npy')
b /= 5


a0 = np.load('frames.npy')
a0_, cnt = fine(a0,0,cnt,b)

a1= np.load('frames1.npy')
a1_, cnt = fine(a1,1,cnt,b)

a2= np.load('frames2.npy')
a2_, cnt = fine(a2,2,cnt,b)

a3= np.load('frames3.npy')
a3_, cnt = fine(a3,3,cnt,b)
x1 = a0_ + a1_ + a2_ + a3_
x1[17] += b[17] - 9.02
cnt[17] += 1
x1[15] += b[15] - 9.72
cnt[15] += 1
x1[17] += b[17] - 9.62
cnt[17] += 1
x1[16] += b[16] - 9.28
cnt[16] += 1
cnt[0]=1
x1 /= cnt

##########################################
cnt = np.zeros(20)
cnt[0]=1
b = np.load('frames_H_fixed_0.npy')
b += np.load('frames_H_fixed_1.npy')
b += np.load('frames_H_fixed_2.npy')
b += np.load('frames_H_fixed_3.npy')
b += np.load('frames_H_fixed_4.npy')
b /= 5


a0 = np.load('frames_0_fix.npy')
a0_, cnt = fine(a0,0,cnt,b)

a1= np.load('frames_1_fix.npy')
a1_, cnt = fine(a1,1,cnt,b)

a2= np.load('frames_2_fix.npy')
a2_, cnt = fine(a2,2,cnt,b)

a3= np.load('frames_3_fix.npy')
a3_, cnt = fine(a3,3,cnt,b)

a4= np.load('frames_4_fix.npy')
a4_, cnt = fine(a4,4,cnt,b)

a5= np.load('frames_5_fix.npy')
a5_, cnt = fine(a5,5,cnt,b)

a6= np.load('frames_6_fix.npy')
a6_, cnt = fine(a6,6,cnt,b)

a7= np.load('frames_7_fix.npy')
a7_, cnt = fine(a7,7,cnt,b)

a8= np.load('frames_8_fix.npy')
a8_, cnt = fine(a8,8,cnt,b)

a9= np.load('frames_9_fix.npy')
a9_, cnt = fine(a9,9,cnt,b)
x2 = a0_ + a1_ + a2_ + a3_ + a4_ + a5_ + a6_ + a7_ + a8_ + a9_

x2 /= cnt
#x1 = -1 * np.load('mine.npy')
s = [i for i in range(1,20)]
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 60,
}
size = 20
wid=3
plt.figure(figsize=(16,16))
plt.plot(s,x1[1:],marker='o',label='stochastic dynamic',linewidth=wid)
plt.plot(s,x2[1:],marker='x',label='deterministic dynamic',linewidth=wid)
plt.tick_params(labelsize=35)
#plt.plot(s,Noise_KSG,label = 'Noisy_KSG',marker='^')
#plt.plot(s,JVHW,label = 'JVHW')
#plt.plot(s,MLE,label = 'MLE')

#plt.xticks(s)
plt.xlabel('Frame Distance',font1)
plt.ylabel('$\mathcal{V}$-information',font1)
plt.legend(fontsize=50)
# plt.show()
plt.savefig('frame.pdf',dpi=300, bbox_inches=None)

