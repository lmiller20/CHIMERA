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

plot_corner = True

# Extract samples:
pic=pickle.load(open('/user/lmiller/CHIMERA_results/OUTPUT_Lauren_new/wasp173_tess_cc_two-limbs-cov_new.pic','rb'), \
                encoding='latin1')

#pic=pickle.load(open('/user/lmiller/CHIMERA_results/OUTPUT_Lauren_test/wasp173_tess_cc_two-limbs-cov.pic','rb'), \
                #encoding='latin1')

samples=pic[:,:-1]

    
with open('tess_bandwidth.txt', 'r') as data:
        tess_w = []
        tess_trans = []
        for line in data:
            p = line.split()
            tess_w.append(float(p[0])) 
            tess_trans.append(float(p[1]))

if plot_corner:
    # Samples has...all the samples. In our case, shape [nsamples, 16]. Let's 
    # extract the common properties of the limbs and plot them in a corner plot:

    # 0: Tirr, 2: log C/O, 3: logKzz, 4: fsed, 5: logPbase, 6:logCldVMR
    samples1 = samples[:, [0,2,3,4,5,6]]

    # 10: Tirr2, 11: log C/O_2, 12: logKzz2, 13: fsed2, 14: logPbase2, 15:logCldVMR2
    samples2 = samples[:, [10, 11, 12, 13, 14, 15]]

    # All right, let's put all this in its own corner plot:
    names = ['T$_{irr}$ (K)', '$\log$ C/O', '$\log K_{zz}$', '$f_{sed}$', '$\log P_{base}$', '$\log$ Cloud$_{VMR}$']
    figure = corner.corner(samples1, color = 'cornflowerblue', labels = names, plot_datapoints='False', plot_contours='False', bins=25, quantiles=[.16,0.5,.84], levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)))
    corner.corner(samples2, fig = figure, color = 'orangered', plot_datapoints='False', plot_contours='False', bins=25, quantiles=[.16,0.5,.84], levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)))
    plt.savefig('plot_comb_tess-cov_wasp173_new.png')
    #plt.savefig('plot_comb_tess-cov_wasp173_test.png')

# Now plot models and spectra. First load cross sections:
if not os.path.exists('model_wavelengths_tess-limbs-cov_wasp173_new.npy'):
#if not os.path.exists('model_wavelengths_tess-limbs-cov_wasp173_test.npy'):
    xsects=xsects_HST(1500,27900) 

# Define function that will evaluate model and return model, depth1 and depth2 at a given set of wavelengths:
def gen_model(cube, wavelengths):

    wlgrid = wavelengths

    Rp = 1.20 #Hellier et al. 2019, Exoplanet Archive #0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
    Rstar = 1.0766100 #TICv8, Exoplanet Archive #0.598   #Stellar Radius in Solar Radii
    M = 3.69 #Hellier et al. 2019, Exoplanet Archive #1.78    #Mass in Jupiter Masses

    #TP profile params (3--Guillot 2010, Parmentier & Guillot 2013--see Line et al. 2013a for implementation)
    #Tirr= 2091 #1500#x[0]#544.54 #terminator **isothermal** temperature--if full redistribution this is equilibrium temp
    #logKir= -1.5  #TP profile IR opacity controlls the "vertical" location of the gradient
    #logg1= -0.7     #single channel Vis/IR opacity. Controls the delta T between deep T and TOA T
    Tint= 200.

    #Composition parameters---assumes "chemically consistnat model" described in Kreidberg et al. 2015
    #logMet= 0.0 #x[1]#1.5742E-2 #.   #Metallicity relative to solar log--solar is 0, 10x=1, 0.1x = -1 used -1.01*log10(M)+0.6
    #logCtoO= -0.26#x[2]#-1.97  #log C-to-O ratio: log solar is -0.26
    logPQCarbon = -5.5  #CH4, CO, H2O Qunech pressure--forces CH4, CO, and H2O to constant value at quench pressure value
    logPQNitrogen = -5.5  #N2, NH3 Quench pressure--forces N2 and NH3 to ""  --ad hoc for chemical kinetics--reasonable assumption

    #A&M Cloud parameters--includes full multiple scattering (for realzz) in both reflected and emitted light
    #logKzz=7 #log Rayleigh Haze Amplitude (relative to H2)
    #fsed=3.0 #haze slope--4 is Rayeigh, 0 is "gray" or flat.  
    #logPbase=-1.0  #gray "large particle" cloud opacity (-35 - -25)
    #logCldVMR=-25.0 #cloud fraction

    #simple 'grey+rayleigh' parameters--non scattering--just pure extinction
    logKcld = -40
    logRayAmp = -30
    RaySlope = 0

    #unpacking parameters to retrieve
    Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp, logKir, logg1 = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],                                                                 cube[6],cube[7], cube[8], cube[9]

    Tirr2, logCtoO2, logKzz2, fsed2 ,logPbase2,logCldVMR2 = cube[10], cube[11], cube[12], cube[13], cube[14], cube[15]

    #print('Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp, logKir, logg1:')
    #print(Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp, logKir, logg1)
    #print('Tirr2, logCtoO2, logKzz2, fsed2 ,logPbase2,logCldVMR2:')
    #print(Tirr2, logCtoO2, logKzz2, fsed2 ,logPbase2,logCldVMR2)
    # Force hemispheres to have different Tirr:
    if Tirr > Tirr2:
        return -np.inf

    ##all values required by forward model go here--even if they are fixed. First, one hemisphere:
    x=np.array([Tirr, logKir,logg1, Tint,logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free
    #wlgrid_fake=np.append(wlgrid,np.array([0.9,1.0,1.2]))
    foo1 = fx_trans(x,wlgrid,gas_scale,xsects)
    #print(foo1[0])
    y_binned1 = foo1[0]
    transmission=tess_trans
   # y_binned_data_1 = (np.nansum(foo1[0]*transmission))/(np.nansum(transmission))

    # Next, the other hemisphere:
    x2=np.array([Tirr2, logKir,logg1, Tint,logMet, logCtoO2, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz2, fsed2,logPbase2,logCldVMR2, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.])
    #print(wlgrid)
    foo2 = fx_trans(x2,wlgrid,gas_scale,xsects)
    #print(foo2[0])
    y_binned2 = foo2[0]
    #y_binned_data_2 = (np.nansum(foo1[0]*transmission))/(np.nansum(transmission))

    y_binned = (y_binned1*0.5) + (y_binned2*0.5)
    
    #print(np.shape(y_binned))
    #print(np.shape(y_binned1*0.5))
    #print(np.shape(y_binned2*0.5))
    
    return y_binned, y_binned1*0.5, y_binned2*0.5
    #return y_binned, y_binned1*0.5, y_binned2*0.5, y_binned_data_1*0.5, y_binned_data_2*0.5

# Load dataset:
"""
wlgrid1, d1, derr1 = np.loadtxt('combined_data_order1.dat',unpack=True)
wlgrid2, d2, derr2 = np.loadtxt('combined_data_order2.dat',unpack=True)
wlgrid = np.append(wlgrid1, wlgrid2)
d = np.append(d1, d2) 
derr = np.append(derr1, derr2)

# Convert:
y_meas = d*1e-6
err = derr*1e-6
idx = np.argsort(wlgrid)
wavelengths, depths, errors = wlgrid[idx], y_meas[idx], err[idx]
"""
#wlgrid1, cl_d1, cl_derr1, hl_d1, hl_derr1,cov1 = np.loadtxt('/home/lmiller/TSRC/final_run/WASP173Ab.dat',unpack=True)
wlgrid1, hl_d1, hl_derr1, cl_d1, cl_derr1,cov1 = np.loadtxt('/home/lmiller/TSRC/final_run/WASP-173Ab_new.dat',unpack=True)
#wlgrid1, hl_d1, hl_derr1, cl_d1, cl_derr1,cov1 = np.loadtxt('/home/lmiller/TSRC/final_run/WASP173Ab.dat',unpack=True)
#wlgrid2, cl_d2, cl_derr2, hl_d2, hl_derr2,cov2 = np.loadtxt('limb_data_order2.dat',unpack=True)
#wlgrid = np.append(wlgrid1, wlgrid2) 
wlgrid = np.array([wlgrid1]) 
#cl_d = np.append(cl_d1, cl_d2)
cl_d = np.array([cl_d1])
#cl_derr = np.append(cl_derr1, cl_derr2)
cl_derr = np.array([cl_derr1])
#hl_d = np.append(hl_d1, hl_d2)
hl_d = np.array([hl_d1])
#hl_derr = np.append(hl_derr1, hl_derr2)
hl_derr = np.array([hl_derr1])

# Convert:
y_meas_cl = cl_d
err_cl = cl_derr

y_meas_hl = hl_d
err_hl = hl_derr

idx = np.argsort(wlgrid)
wavelengths, depths_cl, errors_cl, depths_hl, errors_hl = wlgrid[idx], y_meas_cl[idx], err_cl[idx], y_meas_hl[idx], err_hl[idx]


# If not already done, sample using samples. If already done, just load results:
if os.path.exists('model_wavelengths_tess-limbs-cov_wasp173_new.npy'):
#if os.path.exists('model_wavelengths_tess-limbs-cov_wasp173_test.npy'):

    model_samples = np.load('model_samples_tess-limbs-cov_wasp173_new.npy')
    depth1_samples = np.load('model_depth1_samples_tess-limbs-cov_wasp173_new.npy')
    depth2_samples = np.load('model_depth2_samples_tess-limbs-cov_wasp173_new.npy')
    wavelengths_model = np.load('model_wavelengths_tess-limbs-cov_wasp173_new.npy')
    
#     model_samples = np.load('model_samples_tess-limbs-cov_wasp173_test.npy')
#     depth1_samples = np.load('model_depth1_samples_tess-limbs-cov_wasp173_test.npy')
#     depth2_samples = np.load('model_depth2_samples_tess-limbs-cov_wasp173_test.npy')
#     wavelengths_model = np.load('model_wavelengths_tess-limbs-cov_wasp173_test.npy')

else:
    # Define model wavelengths (ie waelengths at which we'll evaluate model). Here, upper limit is 3 microns because 
    # we want to evaluate up to SOSS wavelengths:   0.36
    #wavelengths_model = np.linspace(np.min(wavelengths), 3.0, 300) 
    wavelengths_model = np.linspace(0.6, 3.0, 264) 

    # Now random sample from the samples, and evaluate model at those random samples:
    nsamples, ndimensions = samples.shape
    nevaluations = 200 # evaluate at 200 random samples
    idx = np.random.choice(np.arange(nsamples),replace=False,size=nevaluations)

    counter = 0
    # Define arrays that will save results:
    
    model_samples, depth1_samples, depth2_samples = np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    np.zeros([nevaluations, len(wavelengths_model)]), \
        
    #model_samples, depth1_samples, depth2_samples, depth1_test_samples, depth2_test_samples = np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    #np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    #np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    #np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    #np.zeros([nevaluations, len(wavelengths_model)]), \
    

    # Sample and evaluate model:
    #print('This is idx:')
    #print(idx)
    for i in idx:
        #print(counter)
        #print('This is i:')
        #print(i)
        model_samples[counter, :], depth1_samples[counter, :], depth2_samples[counter, :] = gen_model(samples[i,:], wavelengths_model)
                #model_samples[counter, :], depth1_samples[counter, :], depth2_samples[counter, :], depth1_test_samples[counter,:], depth2_test_samples[counter,:] = gen_model(samples[i,:], wavelengths_model)
        counter += 1

    # Find nans, correct samples, save:
    idx = np.where(~np.isnan(model_samples[0,:]))[0]
    
    model_samples = model_samples[:,idx]
    depth1_samples = depth1_samples[:,idx]
    depth2_samples = depth2_samples[:,idx]
    wavelengths_model = wavelengths_model[idx]
    #depth1_test_samples = depth1_test_samples[:,idx]
    #depth2_test_samples = depth2_test_samples[:,idx]

    np.save('model_wavelengths_tess-limbs-cov_wasp173_new.npy', wavelengths_model)
    np.save('model_depth1_samples_tess-limbs-cov_wasp173_new.npy', depth1_samples)
    np.save('model_depth2_samples_tess-limbs-cov_wasp173_new.npy', depth2_samples)
    np.save('model_samples_tess-limbs-cov_wasp173_new.npy', model_samples)
    
#     np.save('model_wavelengths_tess-limbs-cov_wasp173_test.npy', wavelengths_model)
#     np.save('model_depth1_samples_tess-limbs-cov_wasp173_test.npy', depth1_samples)
#     np.save('model_depth2_samples_tess-limbs-cov_wasp173_test.npy', depth2_samples)
#     np.save('model_samples_tess-limbs-cov_wasp173_test.npy', model_samples)

# Get 68 and 95 CI for the sampled models:
upper68, lower68 = np.zeros(len(wavelengths_model)), np.zeros(len(wavelengths_model))
upper95, lower95 = np.zeros(len(wavelengths_model)), np.zeros(len(wavelengths_model))
median_model = np.zeros(len(wavelengths_model))

for i in range(len(wavelengths_model)):

    median_model[i], upper68[i], lower68[i] = juliet.utils.get_quantiles(model_samples[:,i], alpha = 0.68)
    _, upper95[i], lower95[i] = juliet.utils.get_quantiles(model_samples[:,i], alpha = 0.95)

# Plot TESS retrieval:
plt.figure(figsize=(6,3))
plt.plot(wavelengths_model, median_model*1e6, color = 'ForestGreen', zorder = 2)
plt.fill_between(wavelengths_model, lower68*1e6, upper68*1e6, color = 'ForestGreen', alpha = 0.2)
plt.fill_between(wavelengths_model, lower95*1e6, upper95*1e6, color = 'ForestGreen', alpha = 0.2)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Transit depth (ppm)')
plt.tight_layout()
plt.savefig('retrieval_tess-limbs-cov_wasp173_new.pdf')
#plt.savefig('retrieval_tess-limbs-cov_wasp173_test.pdf')

# Plot TESS constraint on possible limbs:
plt.figure(figsize=(6,3))
for i in range(model_samples.shape[0]):
    plt.plot(wavelengths_model, depth1_samples[i,:]*1e6, alpha=0.1, color = 'cornflowerblue')
    plt.plot(wavelengths_model, depth2_samples[i,:]*1e6, alpha=0.1, color = 'orangered')
    #plt.plot(wavelengths_model, depth1_test_samples[i,:]*1e6,'o', alpha=0.1, color = 'red')
    #plt.plot(wavelengths_model, depth2_test_samples[i,:]*1e6,'o', alpha=0.1, color = 'yellow')

# Plot data:
plt.errorbar(wavelengths, depths_cl, errors_cl, fmt = 'o', elinewidth=1, ecolor='cornflowerblue', mfc='white', \
             mec='cornflowerblue', zorder = 3)

plt.errorbar(wavelengths, depths_hl, errors_hl, fmt = 'o', elinewidth=1, ecolor='orangered', mfc='white', \
             mec='orangered', zorder = 3)

plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Limb depth (ppm)')
#plt.ylim(5500,7300)
plt.tight_layout()
plt.savefig('tess_limb_constraint-limbs-cov_wasp173_new.pdf')
#plt.savefig('tess_limb_constraint-limbs-cov_wasp173_test.pdf')