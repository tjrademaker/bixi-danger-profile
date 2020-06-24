import pandas as pd
import numpy as np
import csv
import editdistance

def station_location(year):
	"""Reads csv file containing all locations of bixi station for a given year.

	Args:
		year (string)      : year of your bixi data
	Returns:
		station_dict (dict): dictionary of bixi stations
	"""
	# path to station data
	path = 'data/bixi/BixiMontrealRentals'+year+'/Stations_'+year+'.csv'
	# Create dictionary for station names to station codes
	with open(path, mode='r') as f_in:
		reader = csv.reader(f_in)
		station_dict = {rows[1]:rows[0] for rows in reader}
	return station_dict

def filter_bixi_data(df_master):
	"""Parsing through bixi data to remove unnecessary rows, all NaNs from start/end stations.

	Args:
		df_master (pandas.core.frame.DataFrame) : data frame with data from the csv file with user data.
	Returns:
		df (pandas.core.frame.DataFrame)        : filtered data frame

	"""
	# keep only rows that with station name (not timestamps of the day)
	df = df_master.drop(df_master[~(df_master['timestamp'].str.startswith('Start') | df_master['timestamp'].str.startswith('End'))].index)
	df.reset_index(drop = True, inplace = True)
	df.rename(columns = {'timestamp':'date'}, inplace = True)

	# Remove all NaNs from the start/end stations and including its trip-partner (end/start)
	invalid_index = df['location'].isna()
	for index, row in invalid_index.iteritems():
	    if row:
	        if np.mod(index,2) == 1:
	            invalid_index.loc[index-1] = True
	        elif np.mod(index,2) == 0:
	            invalid_index.loc[index+1] = True
	            
	df.drop(df[invalid_index].index,inplace=True)
	return df

def edit_bixi_data(df, station_dict):
	"""For those entries that do not match any key in the dictionary, compute Levenshtein distance for all keys and pick the smallest distance.

	Args:
		df (pandas.core.frame.DataFrame) : data frame with data from the csv file with user data
		station_dict (dict)              : dictionary of bixi stations
	Returns:
		df (pandas.core.frame.DataFrame)        : edited data frame
	"""

	# if entry matches a key in the dictionary
	for index, item in df['location'].iteritems():
	    if item in station_dict.keys():
	        continue
	    else:
	        min_dist = 100
	        # compute Levenshtein distance
	        for station_name in station_dict.keys():
	            dist = editdistance.eval(item, station_name)
	            if dist < min_dist:
	                min_dist = dist
	                min_station = station_name
	        df.loc[index,'location'] = min_station
	return df

def reformat_bixi_data(df, station_dict, filename):
	"""Reformatting data for the rest of the analysis and save file as filename-complete.csv.

	Args:
		df (pandas.core.frame.DataFrame) : data frame with data from the csv file with user data.
		station_dict (dict)              : dictionary of bixi stations	
	Returns:
		df_full (pandas.core.frame.DataFrame)        : reformatted data frame
	"""

	# Create empty dataframe with headers corresponding to known format
	with open('data/bixi/BixiMontrealRentals2018/OD_2018-04.csv', 'r') as f:
	    reader = csv.reader(f)
	    header = next(reader)

	df_full = pd.DataFrame(columns = header)

	# Place the start/end stations and duration in df_full
	for index, row in df.iterrows():
	    if np.mod(index,2) == 0:
	        df_full.loc[int(index/2),'start_station_code'] = station_dict[row[1]]
	        df_full.loc[int(index/2),'duration_sec'] = 60*int(row[2].split()[0])+int(row[2].split()[2])
	    elif np.mod(index,2) == 1:
	        df_full.loc[int(index/2),'end_station_code'] = station_dict[row[1]]

	# save the file
	df_full.to_csv('data/bixi/%s-complete.csv'%filename,index=False)
	return df_full



def filter_rides(df,neighborhood="all",year=2018):
    """Filter rides for which start or end Bixi station (or both) is not within the specified neighborhood. 
    Filters nothing when neighborhood is "all" or not specified
    
    Args:
        df (pd.DataFrame): reformatted dataframe
        neighborhood (str): neighborhood in Montreal that specifies which rides to keep
        year (str): year from which to extract Bixi stations
    Returns:
        df (pd.DataFrame): cleaned dataframe
    """
    
    # Keep all rides
    if neighborhood == "all":
        return df
    
    # List of stations keep rides starting and ending at stations in the Plateau (precomputed boundary)
    elif neighborhood == "Plateau": 
        stations_info=pd.read_csv('data/bixi/BixiMontrealRentals%s/Stations_%s_Plateau.csv'%(year,year))

    # Given a neighborhood name, find Bixi stations in this area
    else:
        stations = create_all_routes(neighborhood,year)
    
    stations=stations_info.code.astype("str").to_list()
    for idx in df.index:
        orig,dest = df.loc[idx,["start_station_code","end_station_code"]]

        # If not both origin and destination station in neighborhood, drop row
        if not ((orig in stations) & (dest in stations)):
            df.drop(idx,axis=0,inplace=True)
    return df
    

### TODO, function is not complete
def create_all_routes(neighborhood,year):
	"""Create routes of specified neighborhood

	Args:
		neighborhood (str): neighborhood in Montreal that specifies stations in what area to keep
		year (str): year from which to extract Bixi stations
	Returns:
		stations (pd.DataFrame): dataframe with station code, intersection, latitude and longitude of station on Bixi stations in a neighborhood during a given year

	TODO:rewrite create-all-routes notebook

	- Takes a while to run
	- open street maps cannot search for neighborhood boundaries just yet, so figure this out

	"""
	stations.to_csv("data/bixi/BixiMontrealRentals%s/Stations_%s_%s.csv"%(year,year,neighborhood))
	return stations



