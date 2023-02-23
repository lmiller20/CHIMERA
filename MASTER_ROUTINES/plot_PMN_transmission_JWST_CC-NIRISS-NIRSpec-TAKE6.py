import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import pickle
import pdb
from fm import *
from scipy import interp
rc('font',family='serif')
import pickle
import corner



#PLOTTING UP CORNER PLOT----------------------------------------------------------------
#import run
pic=pickle.load(open('OUTPUT/pmn_transmission_jwst-FINAL-TAKE6-SOSS-NIRSpec_cc.pic','rb'), encoding='latin1')
samples=pic[:,:-1]
lnprob=pic[:,-1]

outname='pmn_transmission_jwst_cc_TAKE6'


# corner plot
titles=np.array(['T$_{irr}$', '[M/H]', 'log(C/O)', 'log(K$_{zz}$)', 'f$_{sed}$' ,'log(P$_{b}$)','log(VMR$_{cld,b}$)','$\\times$R$_p$','log($P_Q$) C/O', 'log($P_Q$) N'])
priorlow=np.array([  300, -2,  -2,  5.0,  0.5, -6.0, -15., 0.5, -5, -5])
priorhigh=np.array([600, 3.0, 0.3, 11,   6.0,  1.5,  -2,  1.5, 2,2])
Npars=len(titles)
ext=np.zeros([2,Npars])
ext=ext.T
ext[:,0]=priorlow
ext[:,1]=priorhigh
#'''
corner.corner(samples,labels=titles, bins=25,plot_datapoints='False',quantiles=[.16,0.5,.84],show_titles='True',plot_contours='True',extents=ext,levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)))

savefig('./plots/'+outname+"_stair_pairs.pdf",format='pdf')
show()
#'''




#GENERATING RANDOMLY SAMPLES SPECTRA & TP PROFILES-----------------------------------------------------
import numpy as np
xsecs=xsects_JWST(1666,20000)
#xsecs=xsects_JWST(750, 15000)  #5800 cm-1 (1.72 um) to 9100 cm-1 (1.1 um)


Nspectra=200

#loading in data again just to be safe
wlgrid, y_meas, err=np.loadtxt('chimera_toi199-2-sc.dat').T

#setting up default parameter values--SET THESE TO SAME VALUES AS IN LOG-LIKE FUNCTION
#planet/star system params--xRp is the "Rp" free parameter, M right now is fixed, but could be free param
"""
Rp= 0.865#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar=0.829#0.598   #Stellar Radius in Solar Radii
M = 0.271#1.78    #Mass in Jupiter Masses

#TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
Tirr=400. #1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
logKir=-1.5  #TP profile IR opacity controlls the "vertical" location of the gradient
logg1=-0.7     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
Tint=200.

#Composition parameters---assumes "chemically consistnat model" described in Kreidberg et al. 2015
logMet=0.0#x[1]#1.5742E-2 #.   #Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1 used -1.01*log10(M)+0.6
logCtoO=-0.26#x[2]#-1.97  #log C-to-O ratio: log solar is -0.26
logPQCarbon=-5.5  #CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value
logPQNitrogen=-5.5  #N2, NH3 Quench pressure--forces N2 and NH3 to ""  --ad hoc for chemical kinetics--reasonable assumption
#A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
fsed=3.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
logCldVMR=-25.0 #cloud fraction
#simple 'grey+rayleigh' parameters--non scattering--just pure extinction
logKcld = -40
logRayAmp = -30
RaySlope = 0
xRp=1.0
"""
Rp= 0.865#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar=0.829#0.598   #Stellar Radius in Solar Radii
M = 0.271#1.78    #Mass in Jupiter Masses

#TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
Tirr=400. #1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
logKir=-2  #TP profile IR opacity controlls the "vertical" location of the gradient
logg1=-2#-1     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
Tint=300.

#Composition parameters---assumes "chemically consistnat model" described in Kreidberg et al. 2015
logMet=1.0#x[1]#1.5742E-2 #.   #Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1 used -1.01*log10(M)+0.6
logCtoO=-0.26#x[2]#-1.97  #log C-to-O ratio: log solar is -0.26
logPQCarbon= 1.50185314  #CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value
logPQNitrogen= 1.90333421  #N2, NH3 Quench pressure--forces N2 and NH3 to ""  --ad hoc for chemical kinetics--reasonable assumption
#A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
fsed=2.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.
logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
logCldVMR=-2.5 #cloud fraction
#simple 'grey+rayleigh' parameters--non scattering--just pure extinction
logKcld = -38 
logRayAmp = -30 
RaySlope = 0 


#plotting reconstructed TP
draws=np.random.randint(0, samples.shape[0], 500)
logP = np.arange(-6.8,1.5,0.1)+0.1
Tarr=np.zeros((len(draws),len(logP)))
P = 10.0**logP
for i in range(len(draws)):
    Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp,logPQCarbon, logPQNitrogen = samples[draws[i],:]
    #Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp=samples[draws[i],:]
    g0=6.67384E-11*M*1.898E27/(Rp*xRp*71492.*1.E3)**2
    kv=10.**(logg1+logKir)
    kth=10.**logKir
    tp=TP(Tirr, Tint,g0 , kv, kv, kth, 0.5)
    Tarr[i,:] = interp(logP,np.log10(tp[1]),tp[0])


Tmedian=np.zeros(P.shape[0])
Tlow_1sig=np.zeros(P.shape[0])
Thigh_1sig=np.zeros(P.shape[0])
Tlow_2sig=np.zeros(P.shape[0])
Thigh_2sig=np.zeros(P.shape[0])

for i in range(P.shape[0]):
    percentiles=np.percentile(Tarr[:,i],[4.55, 15.9, 50, 84.1, 95.45])
    Tlow_2sig[i]=percentiles[0]
    Tlow_1sig[i]=percentiles[1]
    Tmedian[i]=percentiles[2]
    Thigh_1sig[i]=percentiles[3]
    Thigh_2sig[i]=percentiles[4]

fig, ax=subplots()
fill_betweenx(P,Tlow_2sig,Thigh_2sig,facecolor='r',edgecolor='None',alpha=0.1,label='2-sigma')
fill_betweenx(P,Tlow_1sig,Thigh_1sig,facecolor='r',edgecolor='None',alpha=1.,label='1-sigma')
ax.axis([500,3000,P.max(),P.min()])
ax.semilogy()
plot(Tmedian, P,'b',label='median')
xlabel('Temperature [K]',size='xx-large')
ylabel('Pressure [bar]',size='xx-large')
ax.minorticks_on()
ax.tick_params(length=10,width=1,labelsize='xx-large',which='major')
ax.set_xticks([500, 1500, 2500])
ax.legend(frameon=False,loc=0)
fig.subplots_adjust(left=0.3, right=0.6, top=0.9, bottom=0.1)
savefig('./plots/'+outname+"_TP.pdf",format='pdf')
show()

#'''
#Generating reconstructed spectra by drawing random samples from Posterior
draws=np.random.randint(0, samples.shape[0], Nspectra)
Nwno_bins=xsecs[2].shape[0]
y_mod_array=np.zeros((Nwno_bins, Nspectra))
Reflec_array=np.zeros((Nwno_bins, Nspectra))
Therm_array=np.zeros((Nwno_bins, Nspectra))
y_binned_array=np.zeros((len(wlgrid), Nspectra))

for i in range(Nspectra):
    print(i)
    """
    Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp,logPQCarbon=cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8]

    ##all values required by forward model go here--even if they are fixed
    x=np.array([Tirr, logKir,logg1, Tint,logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free
    foo=fx_trans(x,wlgrid,gas_scale,xsects)
    y_binned=foo[0]

    loglikelihood=-0.5*np.sum((y_meas-y_binned)**2/err**2)  #nothing fancy here
    """
    #make sure this is the same as in log-Like
    Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp,logPQCarbon,logPQNitrogen=samples[draws[i],:]
    x=np.array([Tirr, logKir,logg1, Tint,logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.])
    #wlgrid=-1
    y_binned,y_mod,wno,atm=fx_trans(x,wlgrid,gas_scale, xsecs) 

    y_mod_array[:,i]=y_mod
    y_binned_array[:,i]=y_binned
    
#saving these arrays since it takes a few minutes to generate
pickle.dump([wlgrid, y_meas, err, y_binned_array, wno, y_mod_array],open('./OUTPUT/spectral_samples_trans_pmn_jwst_cc.pic','wb'))
#'''


#PLOTTING SPECTRAL SPREAD-----------------------------------------------------
wlgrid, y_meas, err, y_binned_array, wno, y_mod_array=pickle.load(open('./OUTPUT/spectral_samples_trans_pmn_jwst_cc.pic','rb'))

from matplotlib.pyplot import *
from matplotlib.ticker import FormatStrFormatter

ymax=np.max(y_meas)*1E2*1.02
ymin=np.min(y_meas)*1E2*0.98
fig1, ax=subplots()
xlabel('$\lambda$ ($\mu$m)',fontsize=14)
ylabel('(R$_{p}$/R$_{*}$)$^{2} \%$',fontsize=14)
minorticks_on()

y_median=np.zeros(wno.shape[0])
y_high_1sig=np.zeros(wno.shape[0])
y_high_2sig=np.zeros(wno.shape[0])
y_low_1sig=np.zeros(wno.shape[0])
y_low_2sig=np.zeros(wno.shape[0])

for i in range(wno.shape[0]):
    percentiles=np.percentile(y_mod_array[i,:],[4.55, 15.9, 50, 84.1, 95.45])
    y_low_2sig[i]=percentiles[0]
    y_low_1sig[i]=percentiles[1]
    y_median[i]=percentiles[2]
    y_high_1sig[i]=percentiles[3]
    y_high_2sig[i]=percentiles[4]


cent_lambda, ed = np.loadtxt('no-quench.dat',unpack=True)
plot(cent_lambda, ed, label='No-quenching model', color='grey',zorder=1)

cent_lambda, ed = np.loadtxt('quench.dat',unpack=True)
plot(cent_lambda, ed, label='Quenched (input model)', color='blue',zorder=1)

fill_between(1E4/wno[::-1],y_low_2sig[::-1]*100,y_high_2sig[::-1]*100,facecolor='g',alpha=0.5,edgecolor='None',zorder=0)
fill_between(1E4/wno[::-1],y_low_1sig[::-1]*100,y_high_1sig[::-1]*100,facecolor='g',alpha=0.75,edgecolor='None',zorder=0)
plot(1E4/wno, y_median*100,'g',lw=2,zorder=1)

errorbar(wlgrid, y_meas*100, yerr=err*100, xerr=None, fmt='ok',ms=2, label='Data',alpha=0.5,zorder=2)
ax.set_xscale('log')
ax.set_xticks([0.3, 0.5,0.8,1, 2, 3, 4, 5, 6, 8, 10, 12])
ax.axis([0.5,12,ymin,ymax])

ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(length=5,width=1,labelsize='small',which='major')
savefig('./plots/'+outname+'_spectrum_fits.pdf',fmt='pdf')
show()
close()

# Save outputs for later plotting:
fout = open('final_retrieval_results.dat','w')
fout.write('#wav \t bestfit \t low1sig \t up1sig \t low2sig \t up2sig \n')
for i in range(len(1E4/wno[::-1])):
    fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f} {4:.10f} {5:.10f}\n'.format(1E4/wno[i],y_median[i],y_low_1sig[i],y_high_1sig[i],y_low_2sig[i],y_high_2sig[i]))
fout.close()

