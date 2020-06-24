import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

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