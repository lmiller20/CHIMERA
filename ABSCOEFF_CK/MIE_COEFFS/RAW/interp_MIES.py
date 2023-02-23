import numpy as np
from scipy import interp
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pyplot import *
import pdb
from pickle import *
import h5py

#loading a cross section for wavenumber
file='../../ABSCOEFF_CK/H2H2_CK_STIS_WFC3_10gp_2000_30000wno.h5'
#file='../../ABSCOEFF_CK/CO2_CK_R100_20gp_50_30000wno.h5'

hf=h5py.File(file, 'r')
wno=np.array(hf['wno'])
hf.close()

#loading in Mie file
cond_name='ZnS'  #ALL CONDENSATES
fname_out=cond_name+'_r_0.01_300um_wl_0.3_200um_interp_STIS_WFC3_2000_30000wno.h5'
#fname_out=cond_name+'_r_0.01_300um_wl_0.3_200um_interp_R100_20gp_50_30000wno.h5'
file='./RAW/R1000_Pickles/'+cond_name+'_r_0.01_300um_wl_0.3_200um_R1000'

wlgrid,radius,Mies = load(open(file+'.pic','rb'),  encoding='latin1')

#interpolate
mies_new=np.zeros((3, len(radius), len(wno)))
wno_mie=1E4/wlgrid
for i in range(len(radius)):
    mies_new[0,i,:]=10**np.interp(wno,1E4/wlgrid[::-1],np.log10(Mies[0,i,:][::-1]))
    mies_new[1,i,:]=10**np.interp(wno,1E4/wlgrid[::-1],np.log10(Mies[1,i,:][::-1]))
    mies_new[2,i,:]=10**np.interp(wno,1E4/wlgrid[::-1],np.log10(Mies[2,i,:][::-1]))
    semilogy(1E4/wno, mies_new[0,i,:],'red')
    #semilogy(wlgrid, Mies[0,i,:],'ob')

semilogx()
show()

hf=h5py.File(fname_out,'w')
hf.create_dataset('wno_M',data=wno)
hf.create_dataset('radius',data=radius)
hf.create_dataset('Mies',data=mies_new)
hf.close()


pdb.set_trace()


