import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

datatot = Table.read(f'bphi_results_bphilowS8.txt',format='ascii.no_header')
phs  = ['000','001','002','003','004','005']
b1=[]
zs=[]
bphi=[]
bphidz=[]
# read the file
for tracer in np.unique(datatot['col1']):
    sel= datatot['col1'] == tracer
    data = datatot[sel]
    # the lhs result with different cosmologies
    for zsnap in np.unique(data['col4']):
        b1.append(np.mean(data['col10'][(data['col4']==zsnap)]/2/1.686+1))
        zs.append(zsnap)
        bphi.append(np.mean(data['col7'][(data['col4']==zsnap)]))
    # the rhs results with zsnapshots
    snaps = np.unique(data['col4'])
    for k in range(len(snaps)-1):
        dlog_ndz=np.log(data['col5'][len(phs)*k:len(phs)*(k+1)]/data['col5'][len(phs)*(k+1):len(phs)*(k+2)])
        dlog_sigma8dz = np.log(D(data['col4'][len(phs)*k])/D(data['col4'][len(phs)*(k+1)]))
        bphidz.append(2*dlog_ndz/dlog_sigma8dz)

zmid  = (np.array(zs)[[1,3,4,6,7,8]]+np.array(zs)[[0,2,3,5,6,7]])/2
b1mid = (np.array(b1)[[1,3,4,6,7,8]]+np.array(b1)[[0,2,3,5,6,7]])/2

# compare the bphi-z
plt.plot(np.array(zs),np.array(bphi),'bo',label='LCDM-lowS8 lhs')
plt.plot(zmid,np.array(bphidz),'r^',label='LCDM dz rhs')
plt.legend()
plt.xlabel('redshift')
plt.ylabel('bphi')
plt.savefig('evolution_bphi-z')
plt.close()

# compare the bphi-b1
plt.plot(np.array(b1),np.array(bphi),'bo',label='LCDM-lowS8 lhs')
plt.plot(b1mid,np.array(bphidz),'r^',label='LCDM dz rhs')
plt.plot(np.array(b1),(np.array(b1)-1)*1.686*2,'k--',label=r'$b_{\phi}=2\delta_c(b_1-1)$')
plt.legend()
plt.xlabel('linear bias b1')
plt.ylabel('bphi')
plt.savefig('evolution_bphi-b1')
plt.close()