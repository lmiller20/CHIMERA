import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import pickle
import pdb 
#from fm import *
from scipy import interp
import pickle
import corner
import os
import juliet

plot_corner = False

# Extract samples:
pic=pickle.load(open('/home/lmiller/CHIMERA/MASTER_ROUTINES/OUTPUT_Lauren_test/hp41_wfc3_cc_two-hem.pic','rb'), \
                encoding='latin1')
samples=pic[:,:-1]

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
    plt.savefig('plot_comb.png')

# Now plot models and spectra. First load cross sections:
if not os.path.exists('model_wavelengths.npy'):
    xsects=xsects_HST(1500,27900) 

# Define function that will evaluate model and return model, depth1 and depth2 at a given set of wavelengths:
def gen_model(cube, wavelengths):

    wlgrid = wavelengths

    Rp = 1.685 #0.930#*x[4]# Planet radius in Jupiter Radii--this will be forced to be 10 bar radius--arbitrary (scaling to this is free par)
    Rstar = 1.786 #0.598   #Stellar Radius in Solar Radii
    M = 0.795 #1.78    #Mass in Jupiter Masses

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
    Tirr, logMet, logCtoO, logKzz, fsed ,logPbase,logCldVMR,xRp, logKir, logg1 = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],\
                                                                          cube[6],cube[7], cube[8], cube[9]

    Tirr2, logCtoO2, logKzz2, fsed2 ,logPbase2,logCldVMR2 = cube[10], cube[11], cube[12], cube[13], cube[14], cube[15]

    # Force hemispheres to have different Tirr:
    if Tirr > Tirr2:
        return -np.inf

    ##all values required by forward model go here--even if they are fixed. First, one hemisphere:
    x=np.array([Tirr, logKir,logg1, Tint,logMet, logCtoO, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz, fsed,logPbase,logCldVMR, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.]) #can be made free params if desired (won't affect mmw)#can be made free
    foo1 = fx_trans(x,wlgrid,gas_scale,xsects)
    y_binned1 = foo1[0]

    # Next, the other hemisphere:
    x2=np.array([Tirr2, logKir,logg1, Tint,logMet, logCtoO2, logPQCarbon,logPQNitrogen, Rp*xRp, Rstar, M, logKzz2, fsed2,logPbase2,logCldVMR2, logKcld, logRayAmp, RaySlope])
    gas_scale=np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1., 1., 1.])
    foo2 = fx_trans(x2,wlgrid,gas_scale,xsects)
    y_binned2 = foo2[0]

    y_binned = (y_binned1*0.5) + (y_binned2*0.5)

    return y_binned, y_binned1*0.5, y_binned2*0.5

# Load dataset:
wmin, wmax, d, derr = np.loadtxt('hat-p-41-data.dat',unpack=True)
wlgrid = (wmin + wmax)*0.5
y_meas = d*1e-6
err = derr*1e-6
idx = np.argsort(wlgrid)
wavelengths, depths, errors = wlgrid[idx], y_meas[idx], err[idx]

# If not already done, sample using samples. If already done, just load results:
if os.path.exists('model_wavelengths.npy'):

    model_samples = np.load('model_samples.npy')
    depth1_samples = np.load('model_depth1_samples.npy')
    depth2_samples = np.load('model_depth2_samples.npy')
    wavelengths_model = np.load('model_wavelengths.npy')

else:
    # Define model wavelengths (ie waelengths at which we'll evaluate model). Here, upper limit is 3 microns because 
    # we want to evaluate up to SOSS wavelengths:
    wavelengths_model = np.linspace(np.min(wavelengths), 3.0, 300) 

    # Now random sample from the samples, and evaluate model at those random samples:
    nsamples, ndimensions = samples.shape
    nevaluations = 200 # evaluate at 200 random samples
    idx = np.random.choice(np.arange(nsamples),replace=False,size=nevaluations)

    counter = 0
    # Define arrays that will save results:
    model_samples, depth1_samples, depth2_samples = np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    np.zeros([nevaluations, len(wavelengths_model)]), \
                                                    np.zeros([nevaluations, len(wavelengths_model)])

    # Sample and evaluate model:
    for i in idx:
        model_samples[counter, :], depth1_samples[counter, :], depth2_samples[counter, :] = gen_model(samples[i,:], wavelengths_model)
        counter += 1

    # Find nans, correct samples, save:
    idx = np.where(~np.isnan(model_samples[i,:]))[0]

    model_samples = model_samples[:,idx]
    depth1_samples = depth1_samples[:,idx]
    depth2_samples = depth2_samples[:,idx]
    wavelengths_model = wavelength_models[idx]

    np.save('model_wavelengths.npy', wavelengths_model)
    np.save('model_depth1_samples.npy', depth1_samples)
    np.save('model_depth2_samples.npy', depth2_samples)
    np.save('model_samples.npy', model_samples)

# Get 68 and 95 CI for the sampled models:
upper68, lower68 = np.zeros(len(wavelengths_model)), np.zeros(len(wavelengths_model))
upper95, lower95 = np.zeros(len(wavelengths_model)), np.zeros(len(wavelengths_model))
median_model = np.zeros(len(wavelengths_model))

for i in range(len(wavelengths_model)):

    median_model[i], upper68[i], lower68[i] = juliet.utils.get_quantiles(model_samples[:,i], alpha = 0.68)
    _, upper95[i], lower95[i] = juliet.utils.get_quantiles(model_samples[:,i], alpha = 0.95)

# Plot HST retrieval:
plt.figure(figsize=(6,3))
plt.errorbar(wavelengths, depths*1e6, errors*1e6, fmt = '.', elinewidth=1, ecolor='black', mfc='black', mec='black', zorder = 3)
plt.plot(wavelengths_model, median_model*1e6, color = 'ForestGreen', zorder = 2)
plt.fill_between(wavelengths_model, lower68*1e6, upper68*1e6, color = 'ForestGreen', alpha = 0.2)
plt.fill_between(wavelengths_model, lower95*1e6, upper95*1e6, color = 'ForestGreen', alpha = 0.2)
plt.ylim(9800,11000)
plt.xlim(np.min(wavelengths_model), 1.69)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Transit depth (ppm)')
plt.tight_layout()
plt.savefig('retrieval.pdf')

# Plot HST constraint on possible limbs:
plt.figure(figsize=(6,3))
for i in range(model_samples.shape[0]):
    plt.plot(wavelengths_model, depth1_samples[i,:]*1e6, alpha=0.1, color = 'cornflowerblue')
    plt.plot(wavelengths_model, depth2_samples[i,:]*1e6, alpha=0.1, color = 'orangered')

#plt.ylim(9800,11000)
plt.xlim(np.min(wavelengths_model), 3.0)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Limb depth (ppm)')
plt.tight_layout()
plt.ylim(4900,5600)
plt.savefig('hst_limb_constraint.pdf')
