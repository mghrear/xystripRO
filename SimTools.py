from ROOT import TFile, TVector3, TMath, TTree
from ROOT import TFile
from array import array
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import root_pandas as rp
import pandas as pd
from math import sqrt, fabs
from scipy.optimize import curve_fit


# Mass of electron [MeV/c^2]
m_e = 0.511 


# Radiation length of gases considered [m]
Rad_Lengths = {
    "cf4":89.93,
    "he": 5671,
    "co2": 196.5,
    "ch4": 696.5,
    "c2h6": 361.5,
    "ne": 345.0,
    "ar": 117.6,
    "xe": 15.47,
    "he_cf4":219.6,
    "he_co2": 606.0,
}


MS_fitting_data = pd.read_pickle("/Users/majdghrear/Lab/ang_res_e_gas/sim_ang_res_e_gas/fitting_MS_models/MS_fitting_results.pk")

# Highland equation fit parameters as obtained from fit_models.ipynb
highland_fit_params = {
	"S_2" : MS_fitting_data.iloc[0].Value ,
	"S_2_err" : MS_fitting_data.iloc[1].Value ,
	"Eps" : MS_fitting_data.iloc[2].Value ,
	"Eps_err" : MS_fitting_data.iloc[3].Value 
}

# Rossi equation fit parameters as obtained from fit_models.ipynb
rossi_fit_params = {
	"S_2" : MS_fitting_data.iloc[4].Value ,
	"S_2_err": MS_fitting_data.iloc[5].Value 
}

# Uncertainty calculated on S_2
S2_std = MS_fitting_data.iloc[6].Value


#Defining propoerties of relavent gas mixtures
he_cf4 ={
	'name': 'he_cf4',
	'sigma_T': 136.471, # For He:CF4 use 136.471 MICRONS/CENTIMETER**0.5
	'sigma_r': 100.0 # readout resolution [microns]
}

he_co2 ={
	'name': 'he_co2',
	'sigma_T': 0, # For He:CF4 use 136.471 MICRONS/CENTIMETER**0.5
	'sigma_r': 100.0, # readout resolution [microns]
	'overide_sigma': 466.0 #overide the sigma value with a single entry [microns]
}



#Read raw degrad simulation file (with time ordering data)
def read_degrad(fname):

	# If file doesn't exist return empty lists
	try:
		# List to store 3D positions and corresponding times
		tracks = []
		times = []

		# open root file
		df = rp.read_root(fname)

		# Loop through ROOT tree, get the tracks and corresponding times
		for index, row in df.iterrows():		
			tracks +=  [np.array([row.x,row.y,row.z]).T]
			times += [np.array(row.t)]

		return tracks,times

	except:
		return [],[]


# Order tracks wrt time
def order_track(track,time):

	# Add time coord to each position
	x = np.concatenate((track,time.reshape(len(time),1)),axis=1).astype(float)
	# Order track wrt time
	y = x[x[:,3].argsort()]

	return(y)


#Fit principa axis via SVD
def fitsvd (xpoints, ypoints, zpoints):
    x = np.array(xpoints)
    y = np.array(ypoints)
    z = np.array(zpoints)

    data = np.concatenate((x[:, np.newaxis], 
                           y[:, np.newaxis], 
                           z[:, np.newaxis]), 
                          axis=1)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # dd[0] contains the singlar value corresponding to vv[0], so the RMS
    # _along_ that principal component - i.e. the RMS length

    dir = TVector3(vv[0][0],vv[0][1],vv[0][2])
    return dir, dd, datamean

    

# fit a direction to the track
def get_dir(track,time,dist):

	#order wrt time (not experimentally available)
	ordered_track = order_track(track,time)

	x = []
	y = []
	z = []

	# Loop through time-orderd track keeping only charge in the beginning 
	for vector in ordered_track:

		x += [vector[0]]
		y += [vector[1]]
		z += [vector[2]]

		v = TVector3(0.0,vector[1],vector[2])

		# Stop looping once charge passes beyong fit length 
		if (vector[0]**2 + vector[1]**2 + vector[2]**2) >= (dist**2):
			break

	# This is the direction we want the angle to
	# No diffusion so we use the final point for the direction
	# Almost equivalent results to using SVD for direction.
	Vect = v * (1.0/sqrt(v.Dot(v)))


	return(Vect,x,y,z)

# fit direction to diffused track
def get_dir_diffused(track,time,dist):

	track_T = track.T

	x,y,z = track_T[0], track_T[1], track_T[2]

	select = (x**2+y**2+z**2) < (dist**2)

	y,z =  y[select], z[select]
	x = np.zeros(len(z))

	# Fit the points kept to a principal axis
	Vect, dd, datamean = fitsvd (x, y, z)
	if Vect[2] < 0:
		Vect = -1.0*Vect

	return(Vect,x,y,z)

def get_N_vs_x(track,time,dists):

	#order wrt time (not experimentally available)
	ordered_track = order_track(track,time)

	charge_counts = []

	for dist in dists:

		charge_count = 0 

		# Loop through ordered track
		for vector in ordered_track:

			#increment charge count
			charge_count += 1

			#stop when desired length is reached
			if (vector[0]**2 + vector[1]**2 + vector[2]**2) >= (dist**2):
				break

		charge_counts += [charge_count]


	return charge_counts

# Used to make equal axis when plotting
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


# Gaussian fit function where only parameter is sigma
def Gauss(x,sigma):
	return 1.0/(sigma*sqrt(2*np.pi))*np.e**(-(x**2)/(2*sigma**2))

# Fit a gaussian to an angular distribution
# Creates plot and returns fitted sigma with uncertainty 
def fit_gauss(angles, gas, Energy, dist,  Trim = 2.0):

    # Trim the distribution
	p1 = np.percentile(angles,Trim/2.0)
	p2 = np.percentile(angles,100.0-Trim/2.0)
	angles2 = []
	for g in angles:
		if g > p1 and g < p2:
			angles2 += [g]
	angles = angles2

	# Convert to degrees
	angles = np.array(angles)*180.0/np.pi

	# Bin the data s.t. the center of the central bin is 0,
	# the outermost angle is in the center of the outermost bin
	# and binning is symmetric about 0

	#num is the number of bins on either side of the gaussian
	num = 10
	
	M1 = fabs(min(angles))
	M2 = fabs(max(angles))
	M = max(M1,M2)

	# Make the bin ceneters
	bin_centers = np.arange(-M,M+(M/num),M/num)
	# Number of bin centers must be odd but sometimes numpy gives me and extra bin, I throw it out here
	if len(bin_centers)%2 == 0:
		bin_centers = bin_centers[0:(len(bin_centers)-1)]

	# Make the bin edges
	bin_edges = []
	for center in bin_centers:
		bin_edges += [center - M/(num*2.0)]
	bin_edges += [M + M/(num*2.0)]

	plt.figure()
	# Bin the data
	n, bins, patches = plt.hist(angles, density=True , bins=bin_edges)

	#Ignore bins with no hits
	bin_centers = bin_centers[n != 0]
	diffs = np.diff(bins)[n != 0]
	n = n[n != 0]

	#Calculate number of hits per bin 
	n_hits = n * (len(angles)*diffs)
	#Calculate poission error bars on number of hits
	n_hits_err = np.sqrt(n_hits)
	#calculate error bars on n
	n_err = n * (n_hits_err/n_hits)


	plt.xlabel('Angle [Deg]')
	plt.ylabel('Probability')
	plt.errorbar(bin_centers, n, yerr=n_err,fmt='ko')

	# Fit to binned density function 
	popt, pcov = curve_fit(Gauss, bin_centers, n, sigma=n_err, absolute_sigma=True)
	perr = np.sqrt(np.diag(pcov))


	# Plot the fit 
	xs = np.arange(-M,M,2.0*M/1000)
	plt.plot(xs,Gauss(xs,*popt),'r-',label='fit: $\sigma$ = %5.3f $\pm$ %5.3f, %5.3f percent' % tuple([popt[0],perr[0],perr[0]/popt[0]*100.0]))
	#plt.plot(xs,Gauss(xs,popt[0]+perr[0]),'r--',label='fit Err high')
	#plt.plot(xs,Gauss(xs,popt[0]-perr[0]),'r--',label='fit Err low')
	plt.legend()

	plt.savefig('./plots/'+gas+'_'+str(Energy)+'keV_'+str(Trim)+'per_trim_'+str(dist)+'cm_range.pdf')


	return popt[0], perr[0]

# Highland fitting fuction
# E is energy in MeV, x is distance in m, X_o is Rad length in m, S_2 and eps are fit parameters
def highland_fit(X, S_2, eps):

    E,x,X_o = X

    # Calculate gamma factor 
    Gamma = 1.0 + E/m_e

    # Calculate beta factor
    Beta = np.sqrt(1-(1.0/(Gamma**2)))

    return 57.2958*S_2/(Gamma*(Beta**2)*m_e)*np.sqrt(x/X_o)*(1+eps*np.log(x/(X_o* (Beta**2) )))/np.sqrt(3)

# Rossi fitting function
# E is energy in MeV, x is distance in m, X_o is Rad length in m, S_2 is a fit parameter
def rossi_fit(X,S_2):
    
    E,x,X_o = X

    # Calculate gamma factor 
    Gamma = 1.0 + E/m_e

    # Calculate beta factor
    Beta = np.sqrt(1-(1.0/(Gamma**2)))

    return 57.2958*S_2/(Gamma*(Beta**2)*m_e)*np.sqrt(x/X_o)/np.sqrt(3)



# Formula for RMS due to point resolution
# x is distance in m, dNdx is in 1/cm
# Modified for 3D angles
def Point_Res(x,dNdx,sigma):

    #number of electrons
    N = x * 100 * dNdx

    # L is x in microns
    L = x * 1000000

    sigma_phi = sqrt(12)*sigma/(L*sqrt(N))

    return 57.2958 * sigma_phi

# Analytical calculation of the optmal fit length from rossi_fit and Point_Res
def Opt_len(X,S_2, sigma, dNdx):
        
    E, X_o = X

    # Calculate gamma factor 
    Gamma = 1.0 + E/m_e

    # Calculate beta factor
    Beta = np.sqrt(1-(1.0/(Gamma**2)))
        
    #number of electrons / meter
    N = 100 * dNdx
    
    a = S_2/(Gamma*(Beta**2)*m_e)*np.sqrt(1.0/X_o)/np.sqrt(3)
    
    b = sqrt(12)*sigma/(1000000*sqrt(N))

    return  3.0**(1.0/4.0) * np.sqrt(b/a)




# Fit a line to get the slope
def slope_fit(X, a):
	return a*X



def WriteNtuple(tracks, times, tree_name, root_name):
    """Write root file with ionization distribution. Each individual charge is saved"""


    file = TFile(root_name+'.root', "recreate")


    tree = TTree('tree_'+tree_name, 'tree_'+tree_name)
    
    maxhits     = 20000
    event       = array( 'i', [ 0 ] )
    npoints     = array( 'i', [ 0 ] )

    x = array( 'f', maxhits*[ 0. ] )
    y = array( 'f', maxhits*[ 0. ] )
    z = array( 'f', maxhits*[ 0. ] )
    t = array( 'f', maxhits*[ 0. ] )


    tree.Branch('event_number', event,      'event_number/I')
    tree.Branch('npoints',      npoints,    'npoints/I')
    
    tree.Branch('x',        x,          'x[npoints]/F' )
    tree.Branch('y',        y,          'y[npoints]/F' )
    tree.Branch('z',        z,          'z[npoints]/F' )
    tree.Branch('t',        t,          't[npoints]/F' )

        
    # retrieve each track, fill tree
    
    for track,time in zip(tracks,times):
        if len(track) == 0:
            continue
        index = tracks.index(track)
 
        # ntuple/ reconstructed hit-level quantities

        hitcount =0
            
        for charge,charge_t in zip(track,time):
            x [hitcount] = float (charge.X())
            y [hitcount] = float (charge.Y())
            z [hitcount] = float (charge.Z())
            t [hitcount] = float (charge_t)
            hitcount += 1

        hitcount = len(track)
        npoints[0] = hitcount
        event [0] = index


    

        if hitcount >= maxhits:
            print("   ERROR ! not all hits in tracks were saved. Make this code more clever or increase variable maxhits")
            print("           there were ", hitcount, "quantized charges in track", "but only ", maxhits, "hits were saved.")

        tree.Fill()

    # write root fi

