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
pic=pickle.load(open('OUTPUT/TOI-199_transmission_jwst_free_NEWSOSS.pic','rb'), encoding='latin1')
samples=pic[:,:-1]
lnprob=pic[:,-1]

outname='pmn_transmission_jwst_freeNEWSOSS'

# corner plot
titles=np.array(['T$_{iso}$', 'log(H2O)','log(CO)','log(CO2)','log(CH4)','log(K$_{cld}$)','$\\times$R$_p$'])
priorlow=np.array([  100, -12,  -12,-12 ,-12 ,-45 , 0.5])
priorhigh=np.array([ 800, 0,    0.,  0,   0, -25 , 1.5])
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
xsects=xsects_JWST(1666,20000) # 1666 cm-1 (6 um) to 20000 cm-1 (0.5 um --- to include TESS)
#xsects=xsects_HST(2000, 30000)

Nspectra=200

#loading in data again just to be safe
#wlgrid, y_meas, err=np.loadtxt('w43b_trans.txt').T
wlgrid, y_meas, err = np.loadtxt('chimera_toi199-2.dat',unpack=True)

gelectron, gH2, gH, gHplus,gHminus,gVO,gTiO,gCO2,gHe,gH2O,gCH4,gCO,gNH3,gN2,gPH3,gH2S,gFe,gNa,gK = 0.104E-12, 0.823E+00, 0.262E-07, 0.451E-37, 0.207E-17, 0.793E-18, 0.295E-20, 0.943E-06, 0.163E+00, 0.806E-02, 0.451E-02, 0.129E-03, 0.146E-03, 0.607E-03, 0.394E-06, 0.259E-03, 0.111E-11, 0.664E-05, 0.539E-06
#setting up default parameter values--SET THESE TO SAME VALUES AS IN LOG-LIKE FUNCTION

"""
#planet/star system params--xRp is the "Rp" free parameter, M right now is fixed, but could be free param
Rp= 1.036#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar=0.667#0.598   #Stellar Radius in Solar Radii
M =2.034#1.78    #Mass in Jupiter Masses

#TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
Tirr=1400#1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
logKir=-1.5  #TP profile IR opacity controlls the "vertical" location of the gradient
logg1=-0.7     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
Tint=200.

#A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
fsed=3.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
logCldVMR=-25.0 #cloud fraction

#simple 'grey+rayleigh' parameters--non scattering--just pure extinction
logKcld = -40
logRayAmp = -30
RaySlope = 0

H2O=-15.
CH4=-15.
CO=-15.  
CO2=-15. 
NH3=-15.  
N2=-15.   
HCN=-15.   
H2S=-15.  
PH3=-15.  
C2H2=-15. 
C2H6=-15. 
Na=-15.    
K=-15.   
TiO=-15.   
VO=-15.   
FeH=-15.  
H=-15.     
em=-15. 
hm=-15.
"""

Rp= 0.865#0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar= 0.8292#0.598   #Stellar Radius in Solar Radii
M = 0.271#1.78    #Mass in Jupiter Masses

#TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
Tirr=420 #1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
logKir=-1.5  #TP profile IR opacity controlls the "vertical" location of the gradient
logg1=-0.7     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
Tint=200.

#A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
logKzz=8 #log Rayleigh Haze Amplitude (relative to H2)
fsed=2.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
logCldVMR=-550 #cloud fraction

#simple 'grey+rayleigh' parameters--non scattering--just pure extinction
logKcld = -40 
logRayAmp = -30 
RaySlope = 0 

H2O=np.log10(gH2O)
CH4=np.log10(gCH4)
CO=np.log10(gCO)
CO2=np.log10(gCO2)
NH3=np.log10(gNH3)
N2=np.log10(gN2)
HCN=-15.   
H2S=np.log10(gH2S)  
PH3=np.log10(gPH3)  
C2H2=-15. 
C2H6=-15. 
Na=np.log10(gNa)    
K=np.log10(gK)   
TiO=np.log10(gTiO)   
VO=np.log10(gVO)   
FeH=-15.  
H=np.log10(gH)    
em=np.log10(gelectron)
hm=np.log10(gHminus)

#plotting reconstructed TP
draws=np.random.randint(0, samples.shape[0], 500)
logP = np.arange(-6.8,1.5,0.1)+0.1
Tarr=np.zeros((len(draws),len(logP)))
P = 10.0**logP
for i in range(len(draws)):
    Tirr, H2O,CO,CO2,CH4, logKcld, xRp=samples[draws[i],:]
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
ax.axis([0.5*Tmedian.min(),1.5*Tmedian.max(),P.max(),P.min()])
ax.semilogy()
plot(Tmedian, P,'b',label='median')
xlabel('Temperature [K]',size='xx-large')
ylabel('Pressure [bar]',size='xx-large')
ax.minorticks_on()
ax.tick_params(length=10,width=1,labelsize='xx-large',which='major')
ax.set_xticks([0, 500, 1500, 2500])
ax.legend(frameon=False,loc=0)
fig.subplots_adjust(left=0.3, right=0.6, top=0.9, bottom=0.1)
savefig('./plots/'+outname+"_TP.pdf",format='pdf')
show()

#'''
#Generating reconstructed spectra by drawing random samples from Posterior
draws=np.random.randint(0, samples.shape[0], Nspectra)
Nwno_bins=xsects[2].shape[0]
y_mod_array=np.zeros((Nwno_bins, Nspectra))
Reflec_array=np.zeros((Nwno_bins, Nspectra))
Therm_array=np.zeros((Nwno_bins, Nspectra))
y_binned_array=np.zeros((len(wlgrid), Nspectra))

for i in range(Nspectra):
    print(i)
    #make sure this is the same as in log-Like
    Tirr, H2O,CO,CO2,CH4, logKcld, xRp=samples[draws[i],:]
    x=np.array([Tirr, logKir,logg1, Tint,0,0,0,0, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    print(samples[draws[i],:])
    gas_scale=np.array([H2O,CH4,CO,CO2,NH3,N2,HCN,H2S,PH3,C2H2,C2H6,Na,K,TiO,VO ,FeH,H,-50.,-50.,em, hm,-50.]) 
    y_binned,y_mod,wno,atm=fx_trans_free(x,wlgrid,gas_scale, xsects) 

    y_mod_array[:,i]=y_mod
    y_binned_array[:,i]=y_binned
    
#saving these arrays since it takes a few minutes to generate
pickle.dump([wlgrid, y_meas, err, y_binned_array, wno, y_mod_array],open('./OUTPUT/spectral_samples_trans_pmn_wfc3_free.pic','wb'))
#'''


#PLOTTING SPECTRAL SPREAD-----------------------------------------------------
wlgrid, y_meas, err, y_binned_array, wno, y_mod_array=pickle.load(open('./OUTPUT/spectral_samples_trans_pmn_wfc3_free.pic','rb'))

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


fill_between(1E4/wno[::-1],y_low_2sig[::-1]*100,y_high_2sig[::-1]*100,facecolor='g',alpha=0.5,edgecolor='None',zorder=0)
fill_between(1E4/wno[::-1],y_low_1sig[::-1]*100,y_high_1sig[::-1]*100,facecolor='g',alpha=0.75,edgecolor='None',zorder=0)
plot(1E4/wno, y_median*100,'g',zorder=3)


errorbar(wlgrid, y_meas*100, yerr=err*100, xerr=None, fmt='Dk',zorder=3)
ax.set_xscale('log')
ax.set_xticks([0.3, 0.5,0.8,1,1.4, 2, 3, 4, 5])
ax.axis([0.3,5.0,ymin,ymax])

#wm1,mm1 = np.loadtxt('g9_10X_quenched.txt',unpack=True)
#wm2,mm2 = np.loadtxt('g9_10X_non-quenched.txt',unpack=True)
#from scipy.ndimage import gaussian_filter1d
#plot(wm1,gaussian_filter1d(mm1*100,6),color='cornflowerblue',zorder=2)
#plot(wm2,gaussian_filter1d(mm2*100,6),color='orangered',zorder=2)

ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(length=5,width=1,labelsize='small',which='major')
savefig('./plots/'+outname+'_spectrum_fits.pdf',fmt='pdf')
show()
close()


