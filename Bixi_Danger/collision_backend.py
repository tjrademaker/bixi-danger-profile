import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

def coordinate_grid(latmin, latmax, lonmin, lonmax, nlat, nlon):
	"""Creating a grid of latitude and longitude coordinates.

	Args:
		latmin (float) : latitude lower boundary
		latmax (float) : latitude upper boundary
		lonmin (float) : longitude lower boundary
		lonmax (float) : longitude upper boundary
		nlat   (int)   : number of latitude sample
		nlon   (int)   : number of longitude sample		
	Returns:
		delta  (tuple) : latitude and longitude step sizes
		position (ndarray) : grid of coordinates
		mins   (1D-array) : latitude and longitude lower boudaries
		maxs   (1D-array) : latitude and longitude upper boudaries
	"""

	# Grouping boundaries
	mins, maxs = np.array([latmin,lonmin]),np.array([latmax,lonmax])
	# Creating meash grid
	x = np.linspace(latmin, latmax, nlat)
	y = np.linspace(lonmin, lonmax, nlon)
	xv, yv = np.meshgrid(x, y)

	delta = np.array([x[1]-x[0],y[1]-y[0]])
	position = np.dstack((xv, yv))
	return delta, position, mins, maxs

def get_density_map(year, nstep = 1000):
	"""Create coordinate map and marks how mane bike traversed each coordinates based on BIXI data.

	Args:
		year (str) : year of BIXI data used
		nstep (int) : nstep in the map grid
	Returns:
		density_map  (ndarray) : traffic density grid map

	"""
	# create empty map
	density_map = np.zeros((nstep, nstep))

	# open all bixi routes pickle
	for k in range(4,11):
		print('month is %d'%k)
		month = str(k)
		df = pd.read_csv('data/bixi/BixiMontrealRentals'+year+'/OD_'+year+'-0'+month+'-filtered.csv')
		for j in range(len(df)):
			try:
				route = dic[df['name'].values[j]]
				route = (np.array(route) - mins)/delta
				addition = []
				# Make it such that the largest difference (either in x or in y) per consecutive datapoint makes jumps of 1
				for i in range(len(route)-1):
					diff = (route[i+1,0]-route[i,0],route[i+1,1]-route[i,1])
					max_diff = np.max(diff)
					[addition.append([route[i,0]+max_diff*j/diff[0],route[i,1]+max_diff*j/diff[1]]) for j in range(int(max_diff))]
				coords = np.concatenate((route,np.array(addition)))
				# add count to density map
				for coord in coords:
					if ((coord[0] > 1000) | (coord[1] > 1000)):
						continue
					else:
						density_map[int(coord[0]),int(coord[1])] += 1
			except:
				continue
	return density_map

def save_maps(density_map):
	"""Create boolean map for de density map and saves the maps

	Args:
		density_map  (ndarray) : traffic density grid map

	Returns:
		boolean_map  (ndarray) : boolean traffic density grid map

	"""
	boolean_map = density_map.copy()
	boolean_map[boolean_map != 0] = 1
	with open('../data/maps.pkl','wb') as f:
		pickle.dump((density_map,boolean_map),f)
	return boolean_map

def filter_collision(latmin, latmax, lonmin, lonmax):
	"""Create boolean map for de density map and saves the maps

	Args:
		latmin (float) : latitude lower boundary
		latmax (float) : latitude upper boundary
		lonmin (float) : longitude lower boundary
		lonmax (float) : longitude upper boundary	
	Returns:
		lat_short (1D array): filtered latitutes
		lon_short (1D array): filtered longitudes
	"""

	path = 'data/cluster_means.csv'
	# load accident data [lan, lon, weight]
	data = np.loadtxt(path, usecols=([1,2,3]), skiprows=1, delimiter=',')

	lat, lon, weight = data[:,0], data[:,1], data[:,2] 

	# get index of value outside of latitude range
	ind1 = np.where(latmin>lat)
	ind2 = index1 = np.where(lat >latmax)

	# get index of value outside of longitude range
	ind3 = np.where(lonmin>lon)
	ind4 = index1 = np.where(lon>lonmax)

	# merge indexes
	ind  = np.unique(np.concatenate((ind1, ind2, ind3, ind4), axis=None))

	lon_short = np.delete(lon,ind)
	lat_short = np.delete(lat,ind)
	weight_short = np.delete(weight, ind)
	return lat_short, lon_short, weight_short

def Gaussian2D(position, amp, xo, yo, sigx=0.005, sigy=0.005):
	'''Create a 2D Gaussian array.
	Args:
		position  (3D array): Meshgrids of x and y indices of pixels. position[:,:,0] = x and position[:,:,1] = y.
		amp	         (float): Amplitude of the 2D Gaussian.
		xo	         (float): x value of the peak of the 2D Gaussian.
		yo	         (float): y value of the peak of the 2D Gaussian.
		sigx         (float): Width of the 2D Gaussian along the x axis.
		sigy         (float): Width of the 2D Gaussian along the y axis.

    Returns:
	    PSF.ravel (1D array):z values of the 2D Gaussian raveled.
	'''
	centroid = [yo, xo]
	cov = [[sigy**2, 0],[0, sigx**2]]
	rv = st.multivariate_normal(mean = centroid, cov = cov)
	PSF = amp*(rv.pdf(position))
	return PSF

def gaussian_danger_map(colli_lat, colli_lon, weight, latmin, latmax, lonmin, lonmax):
	# creating the position
	nx, ny = (1000, 1000)
	x = np.linspace(45.5049722, 45.5412594, nx)
	y = np.linspace(-73.6120455, -73.5593741, ny)
	xv, yv = np.meshgrid(x, y)

	position = np.dstack((xv, yv))

	danger_map = np.zeros((1000, 1000))
	for i in range(len(colli_lon)):
	#for i in range(2):
		tmp = Gaussian2D(position, weight[i], colli_lon[i], 
			colli_lat[i], sigx=0.001, sigy=0.001)
		danger_map += tmp
	return danger_map