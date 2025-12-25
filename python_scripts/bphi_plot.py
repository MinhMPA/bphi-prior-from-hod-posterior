import numpy as np
import matplotlib.pyplot as plt

snaps  = [0.5,0.8,1.09,1.1,1.325,1.11,1.325,1.85,2.5]
bphis  = [2.96,3.63,4.67,1.22,1.75,3.31,3.42,5.58,7.97]
errors = [0.01,0.01,0.02,0.01,0.02,0.06,0.07,0.11,0.08]
ps     = [0.98,1,1.04,0.96,0.98,1.03,1.035,1.065,1.16]
perrs  = [0.01]*5+[0.11,0.037,0.05,0.04]
biases = [1.86,2.08,2.42,1.32,1.50,2.03,2.05,2.70,3.5]
berrs  = [0.01,0.01,0.01,0.01,0.01,0.04,0.02,0.06,0.04]
colors = ['r']*3+['b']*2+['c']*4

plt.rc('font', family='serif', size=12) 
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.15,right = 0.98,bottom=0.15,top = 0.93, )
for i in range(len(snaps)):
    plt.errorbar(snaps[i], bphis[i], yerr=errors[i],
                 fmt='o', color=colors[i], ecolor=colors[i])
plt.errorbar([],[],[],label='LRG',fmt='o',color='r',ecolor='r')
plt.errorbar([],[],[],label='ELG',fmt='o',color='b',ecolor='b')
plt.errorbar([],[],[],label='QSO',fmt='o',color='c',ecolor='c')
plt.xlabel('redshift')
plt.ylabel(r'$b_{\phi}$')
plt.legend()
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('evol_bphi')
plt.close()
   
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.2,right = 0.98,bottom=0.15,top = 0.93, )
for i in range(len(snaps)):
    plt.errorbar(snaps[i], ps[i], yerr=perrs[i],
                 fmt='o', color=colors[i], ecolor=colors[i])
plt.errorbar([],[],[],label='LRG',fmt='o',color='r',ecolor='r')
plt.errorbar([],[],[],label='ELG',fmt='o',color='b',ecolor='b')
plt.errorbar([],[],[],label='QSO',fmt='o',color='c',ecolor='c')
plt.xlabel('redshift')
plt.ylabel('p')
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('evol_p_value')
plt.close()
   
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.15,right = 0.98,bottom=0.15,top = 0.93, )
for i in range(len(snaps)):
    plt.errorbar(biases[i], bphis[i], xerr=berrs[i], yerr=errors[i],
                 fmt='o', color=colors[i], ecolor=colors[i])
plt.errorbar([],[],[],label='LRG',fmt='o',color='r',ecolor='r')
plt.errorbar([],[],[],label='ELG',fmt='o',color='b',ecolor='b')
plt.errorbar([],[],[],label='QSO',fmt='o',color='c',ecolor='c')
plt.plot(biases,(np.array(biases)-1)*1.686*2,'k--',label=r'$b_{\phi}=2\delta_c(b_1-1)$')
plt.xlabel(r'$b_1$')
plt.ylabel(r'$b_{\phi}$')
plt.legend()
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('b1_bphi')
plt.close()