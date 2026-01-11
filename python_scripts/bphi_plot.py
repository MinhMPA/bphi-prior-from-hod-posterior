import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

data = Table.read('bphi_results.txt',format='ascii.no_header')

snaps  = [0.5,0.8,1.09,1.1,1.325,1.11,1.325,1.85,2.5]
bphis  = [2.96,3.63,4.67,1.22,1.75,3.31,3.42,5.58,7.97]
errors = [0.01,0.01,0.02,0.01,0.02,0.06,0.07,0.11,0.08]
ps     = [0.98,1,1.04,0.96,0.98,1.03,1.035,1.065,1.16]
perrs  = [0.01]*5+[0.11,0.037,0.05,0.04]
biases = [1.86,2.08,2.42,1.32,1.50,2.03,2.05,2.70,3.5]
berrs  = [0.01,0.01,0.01,0.01,0.01,0.04,0.02,0.06,0.04]
tracers=['LRG','ELG','QSO']
colors = ['r','b','c']

plt.rc('font', family='serif', size=12) 
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.15,right = 0.98,bottom=0.15,top = 0.93, )
for tracer,color in zip(tracers,colors):
    sel= data['col1'] == tracer
    plt.errorbar(np.mean(data[sel]['col4'].reshape(np.sum(sel)//6,6),axis=1), np.mean(data[sel]['col7'].reshape(np.sum(sel)//6,6),axis=1), yerr=np.std(data[sel]['col7'].reshape(np.sum(sel)//6,6),axis=1),
                        fmt='o', color=color, ecolor=color)
    plt.errorbar([],[],[],label=tracer,fmt='o',color=color,ecolor=color)
plt.xlabel('redshift')
plt.ylabel(r'$b_{\phi}$')
plt.legend()
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('evol_bphi')
plt.close()
   
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.2,right = 0.98,bottom=0.15,top = 0.93, )
for tracer,color in zip(tracers,colors):
    sel= data['col1'] == tracer
    plt.errorbar(np.mean(data[sel]['col4'].reshape(np.sum(sel)//6,6),axis=1), np.mean(data[sel]['col8'].reshape(np.sum(sel)//6,6),axis=1), yerr=np.std(data[sel]['col8'].reshape(np.sum(sel)//6,6),axis=1),
                        fmt='o', color=color, ecolor=color)
    plt.errorbar([],[],[],label=tracer,fmt='o',color=color,ecolor=color)
plt.xlabel('redshift')
plt.ylabel('p')
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('evol_p_value')
plt.close()
   
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(left = 0.15,right = 0.98,bottom=0.15,top = 0.93, )
for tracer,color in zip(tracers,colors):
    sel= data['col1'] == tracer
    plt.errorbar(np.mean(data[sel]['col10'].reshape(np.sum(sel)//6,6),axis=1)/2/1.686+1, np.mean(data[sel]['col7'].reshape(np.sum(sel)//6,6),axis=1), xerr=np.std(data[sel]['col10'].reshape(np.sum(sel)//6,6),axis=1)/2/1.686, yerr=np.std(data[sel]['col7'].reshape(np.sum(sel)//6,6),axis=1),
                        fmt='o', color=color, ecolor=color)
    plt.errorbar([],[],[],label=tracer,fmt='o',color=color,ecolor=color)
plt.plot(biases,(np.array(biases)-1)*1.686*2,'k--',label=r'$b_{\phi}=2\delta_c(b_1-1)$')
plt.xlabel(r'$b_1$')
plt.ylabel(r'$b_{\phi}$')
plt.legend()
plt.title('separate Universe with AbacusHOD: c000 vs c004    ',fontsize=10)
plt.savefig('b1_bphi')
plt.close()