import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import pickle
import pdb 
from fm import *
from scipy import interp
import pickle
import corner
import os
import juliet

names = ['tess-limbs']
files = ['/user/lmiller/CHIMERA_results/OUTPUT_Lauren_new/wasp79_tess_cc_two-limbs-cov_new.pic']
plot_corner = False


Rp = 1.67 #Stassun et al. 2017, Exoplanet Archive #0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
Rstar = 1.4853200 #TICv8, Exoplanet Archive #0.598   #Stellar Radius in Solar Radii
M = 0.850 #Hellier et al. 2019, Exoplanet Archive #1.78    #Mass in Jupiter Masses

Tint = 200.
ndraws = 1000

for j in range(len(names)):

    fig, ax = plt.subplots(figsize=(5,8))
    # Extract samples:
    pic=pickle.load(open(files[j],'rb'), encoding='latin1')
    samples=pic[:,:-1]

    print(names[j],samples.shape)

    draws=np.random.randint(0, samples.shape[0], ndraws)
    logP = np.arange(-6.8,1.5,0.1)+0.1
    Tarr1 = np.zeros((len(draws),len(logP)))
    Tarr2 = np.zeros((len(draws),len(logP)))
    P = 10.0**logP

    for i in range(len(draws)):

        Tirr, logMet, logCtoO, logKzz, fsed, logPbase, logCldVMR, xRp, logKir, logg1 = samples[draws[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        Tirr2, logCtoO2, logKzz2, fsed2 ,logPbase2,logCldVMR2 = samples[draws[i], [10, 11, 12, 13, 14, 15]] 

        g0 = 6.67384E-11*M*1.898E27/(Rp*xRp*71492.*1.E3)**2
        kv = 10.**(logg1+logKir)
        kth = 10.**logKir
        tp1 = TP(Tirr, Tint, g0 , kv, kv, kth, 0.5)
        tp2 = TP(Tirr2, Tint, g0 , kv, kv, kth, 0.5)

        Tarr1[i,:] = interp(logP,np.log10(tp1[1]),tp1[0])
        Tarr2[i,:] = interp(logP,np.log10(tp2[1]),tp2[0])

    Tmedian=np.zeros(P.shape[0])
    Tlow_1sig=np.zeros(P.shape[0])
    Thigh_1sig=np.zeros(P.shape[0])
    Tlow_2sig=np.zeros(P.shape[0])
    Thigh_2sig=np.zeros(P.shape[0])

    T2median=np.zeros(P.shape[0])
    Tlow2_1sig=np.zeros(P.shape[0])
    Thigh2_1sig=np.zeros(P.shape[0])
    Tlow2_2sig=np.zeros(P.shape[0])
    Thigh2_2sig=np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        percentiles=np.percentile(Tarr1[:,i],[4.55, 15.9, 50, 84.1, 95.45])
        Tlow_2sig[i]=percentiles[0]
        Tlow_1sig[i]=percentiles[1]
        Tmedian[i]=percentiles[2]
        Thigh_1sig[i]=percentiles[3]
        Thigh_2sig[i]=percentiles[4]

        percentiles2=np.percentile(Tarr2[:,i],[4.55, 15.9, 50, 84.1, 95.45])
        Tlow2_2sig[i]=percentiles2[0]
        Tlow2_1sig[i]=percentiles2[1]
        T2median[i]=percentiles2[2]
        Thigh2_1sig[i]=percentiles2[3]
        Thigh2_2sig[i]=percentiles2[4]

    plt.fill_betweenx(P,Tlow_2sig,Thigh_2sig,facecolor='cornflowerblue',edgecolor='None',alpha=0.3)#,label='2-sigma')
    plt.fill_betweenx(P,Tlow_1sig,Thigh_1sig,facecolor='cornflowerblue',edgecolor='None',alpha=0.5)#,label='1-sigma')
    plt.plot(Tmedian, P, color='cornflowerblue')

    plt.fill_betweenx(P,Tlow2_2sig,Thigh2_2sig,facecolor='orangered',edgecolor='None',alpha=0.3)#,label='2-sigma')
    plt.fill_betweenx(P,Tlow2_1sig,Thigh2_1sig,facecolor='orangered',edgecolor='None',alpha=0.5)#,label='1-sigma')
    plt.plot(T2median, P, color='orangered')

    ax.axis([700,2500,P.max(),P.min()])
    ax.semilogy()
    plt.xlabel('Temperature (K)',size='xx-large')
    plt.ylabel('Pressure (bar)',size='xx-large')
    ax.minorticks_on()
    ax.tick_params(length=10,width=1,labelsize='xx-large',which='major')
    #ax.set_xticks([800, 1200, 1600, 2000])
    plt.tight_layout()
    plt.savefig(names[j]+'_tp_wasp79_new.pdf')
#ax.legend(frameon=False,loc=0)
