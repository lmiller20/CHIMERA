#import matplotlib as mpl
#mpl.use('TkAgg')
import numpy as np
from scipy import *
#from matplotlib.pyplot import *
from pickle import *
from pymiecoated import Mie
import pdb

fname='MgSiO3_mie_r_0.01_300um_wl_0.3_200um.pic'
#load index--wl, n, k
wl,n0,k0=loadtxt('../Indicies_of_Refrac_Wakeford/MgSiO3_complex.txt',skiprows=2).T


#set up wlgrid and interpolate index to wlgrid
wlgrid=np.arange(0.3,200,0.1)
nn=interp(wlgrid,wl,n0)
kk=interp(wlgrid,wl,k0)

#computing properties--looping over wavelength and r
radius=10**(np.arange(-2,2.6,0.1))
Mie_properties=np.zeros((3,len(radius),len(wlgrid)))
#0-qext, 1-qsca, 2-asym  radius x wlgrid
for i in range(len(radius)):
	for j in range(len(wlgrid)):
		xx=2.*np.pi*radius[i]/wlgrid[j]
		mie=Mie(x=xx, m=complex(nn[j],kk[j]))
		qext=mie.qext()
		qsca=mie.qsca()
		asym=mie.asy()
		Mie_properties[0,i,j]=qext #qext
		Mie_properties[1,i,j]=qsca #qsca
		Mie_properties[2,i,j]=asym #asym
		print radius[i],wlgrid[j],qext,qsca,asym
loc=np.where(Mie_properties < 0)
Mie_properties[loc]=1E-100
Mie_properties[isnan(Mie_properties)]=1E-100
#plotting
for i in range(len(radius)):
	plot(wlgrid, Mie_properties[0,i,:]*np.pi*radius[i]**2)

semilogy()
semilogx()
axis([0.3,20,1E-15,1000])
output=[wlgrid, radius,Mie_properties]
dump(output, open(fname,'wb'))

pdb.set_trace()
