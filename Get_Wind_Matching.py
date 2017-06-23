from __future__ import division
import datetime
import time
import sys
import math
import numpy as np
import pickle
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Proj, Geod
import os
from collections import OrderedDict
import zipfile
from dateutil.parser import parse
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
	from cStringIO import StringIO
except:
	from io import BytesIO as StringIO


global BaseTime 
BaseTime = parse('2013-01-01 00:00:00')


# Useful Functions
# Projections
wgs84=Proj("+init=EPSG:4326")
epsg3857=Proj("+init=EPSG:3857")
g=Geod(ellps='WGS84')

def press(alt):
    z = alt/3.28084
    return 1013.25*(1-(0.0065*z)/288.15)**5.255

def proxilvl(alt , lvls):
    p = press(alt)
    levels = np.array(sorted(lvls))
    return levels[np.abs(levels - p).argmin()]

def GetAzimuth(CenterTraj1):
    CenterTraj = CenterTraj1.copy()
    CenterTraj['azimuth'] = 0.0
    FID = []
    for rowid, row in CenterTraj.iterrows():
        if int(row.FID) not in FID:
            FID.append(int(row.FID))
            latl = row.Lat
            lonl = row.Lon
        else:
            CenterTraj.loc[rowid,'azimuth'] = g.inv(lonl, latl, row.Lon, row.Lat)[0]
            latl = row.Lat
            lonl = row.Lon
    return CenterTraj

def DecodeWind(raw_wind_file_loaction = os.getcwd() + '/raw/'):
	# This step requires pygrib installed
	# Only this step needs to be done in a Linux machine, others can be finished either on a windows or on a Linux

	# raw_wind_file_location is a directory that stores only the raw wind data for one year
	import pygrib

	Time = []
	try:
	    os.mkdir('US_wind')
	except:
	    pass

	for rawfname in os.listdir(raw_wind_file_loaction):
	    print(rawfname)
	    grbs = pygrib.open(raw_wind_file_loaction + rawfname)
	    uin = grbs.select(shortName='u', typeOfLevel='isobaricInhPa', level = lambda l: l >= 100 and l <= 1000)
	    vin = grbs.select(shortName='v', typeOfLevel='isobaricInhPa', level = lambda l: l >= 100 and l <= 1000)
	    grbs.close()

	    for i in range(len(uin)):
	        u_wind_i = uin[i]
	        v_wind_i = vin[i]

	        lat_grid = u_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[1].flatten()
	        lon_grid = u_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[2].flatten()
	        lon_lat_grid = np.dstack((lon_grid, lat_grid))[0]

	        pickle.dump(lon_lat_grid, open('LonLat_Grid.p','wb'), protocol=2)

	        date_time = str(u_wind_i.year) + '_' + str(u_wind_i.month).zfill(2) + '_' + str(u_wind_i.day).zfill(2) + '_' + str(u_wind_i.hour).zfill(2) + '00'
	        if date_time not in Time:

	            wind_data = OrderedDict()
	            Time.append(date_time)

	            wind_data[u_wind_i.level] = np.dstack((u_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[0].flatten(),
	                                                     v_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[0].flatten()))[0]

	        else:
	            wind_data[u_wind_i.level] = np.dstack((u_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[0].flatten(),
	                                                     v_wind_i.data(lat1 = 20, lat2 = 55, lon1 = 230, lon2 = 300)[0].flatten()))[0]

	        pickle.dump(wind_data, open(os.getcwd() + '/US_wind/' + date_time + '.p','wb'), protocol=2) # m/s

	with zipfile.ZipFile('NCAR_wind.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
		for fname in os.listdir(os.getcwd() + '/US_wind/'):
			zf.write(os.getcwd() + '/US_wind/' + fname, os.path.basename(os.getcwd() + '/US_wind/' + fname))


def PrepareWind(wind_file = os.getcwd() + '/WIND_NCAR/wind2.zip', grid_file = os.getcwd() + '/WIND_NCAR/LonLat_Grid.p', Save = True):
	# Prepare wind data
	# Construct temporal kdtree (for temporal matching)
	print('---------------- Preparing wind data and constructing kd-trees----------------')
	global BaseTime
	u_wind = []
	v_wind = []
	TimeIdx = []
	idx = -1
	with zipfile.ZipFile(wind_file, "r") as zfile: # Within the zip file
	    file_list = zfile.namelist()
	    for fname in file_list:
	        wind_data = pickle.load(StringIO(zfile.read(fname)))
	        if len(wind_data.keys()) != 12:
	        	print("The file %s is skipped (# of levels ~= 12)" %fname)
	        	pass
	        for key in wind_data.keys():
	            idx += 1

	            dt = (parse(fname[:4]+'-'+fname[5:7]+'-'+fname[8:10]+' '+fname[11:15]) - BaseTime)
	            dt = dt.days * 24. + dt.seconds/3600.
	            if dt not in TimeIdx:
	                TimeIdx.append(dt)
	            else:
	                pass
	            u_wind.append(wind_data[key][:,0])
	            v_wind.append(wind_data[key][:,1])
	levels = sorted(list(wind_data.keys()), reverse = True)
	u_wind = np.array(u_wind)
	v_wind = np.array(v_wind)
	TimeIdxTree = cKDTree(np.array(TimeIdx).reshape(-1,1))

	# Construct spatial kdtree (for spatial matching)
	Lon_Lat_Grid = pickle.load(open(grid_file,'rb'))
	Lon_Lat_Grid[:,0] = Lon_Lat_Grid[:,0] - 360
	Grid_KDTree = cKDTree(Lon_Lat_Grid)
	if Save:
		pickle.dump(u_wind, open('u_wind.p', 'wb'), protocol = 2)
		pickle.dump(v_wind, open('v_wind.p', 'wb'), protocol = 2)
		pickle.dump(TimeIdxTree, open('TimeIdxTree.p', 'wb'), protocol = 2)
		pickle.dump(Grid_KDTree, open('Grid_LonLat_Tree.p', 'wb'), protocol = 2)
		pickle.dump(levels, open('Levels.p', 'wb'), protocol = 2)
		print('Files dumped to %s', os.getcwd())
	print('Finished')
	return u_wind, v_wind, TimeIdxTree, Grid_KDTree, levels, Lon_Lat_Grid


class WindMatching:
	def __init__(self, DEP, ARR, Year, Wind_Preload = False, file_location_dict = {'u_wind':'u_wind.p', 
							  'v_wind': 'v_wind.p', 
							  'TimeIdxTree': 'TimeIdxTree.p', 
							  'Grid_LonLat_Tree':'Grid_LonLat_Tree.p',
							  'Levels': 'Levels.p',
							  'wind_file': os.getcwd() + '/WIND_NCAR/wind2.zip',
							  'grid_file': os.getcwd() + '/WIND_NCAR/LonLat_Grid.p'
							  }):
		# file_location_dict = {'u_wind':'u_wind.p', 
		# 					  'v_wind': 'v_wind.p', 
		# 					  'TimeIdxTree': 'TimeIdxTree.p', 
		# 					  'Grid_LonLat_Tree':'Grid_LonLat_Tree.p',
		# 					  'wind_file': os.getcwd() + '/WIND_NCAR/wind2.zip',
		# 					  'grid_file': os.getcwd() + '/WIND_NCAR/LonLat_Grid.p'
		# 					  }
		global BaseTime
		self.DEP = DEP
		self.ARR = ARR
		self.Year = Year

		if Wind_Preload:
			print('---------------- Preload preprocessed wind files ----------------')
			self.u_wind = pickle.load(open(file_location_dict['u_wind'], 'rb'))
			self.v_wind = pickle.load(open(file_location_dict['v_wind'], 'rb'))
			self.TimeIdxTree = pickle.load(open(file_location_dict['TimeIdxTree'], 'rb'))
			self.Grid_KDTree = pickle.load(open(file_location_dict['Grid_LonLat_Tree'], 'rb'))
			self.Levels = pickle.load(open(file_location_dict['Levels'], 'rb'))
			self.Grid = pickle.load(open(file_location_dict['grid_file'], 'rb'))
			self.Grid[:,0] = self.Grid[:,0] - 360.
			print('Finished')
		else:
			self.u_wind, self.v_wind, self.TimeIdxTree, self.Grid_KDTree, self.Levels, self.Grid = PrepareWind(file_location_dict['wind_file'], 
																					   file_location_dict['grid_file'],
																					   Save = True)
		self.level_idx_dict = {}
		i = -1
		for key in self.Levels:
		    i += 1
		    self.level_idx_dict[key] = i

		self.LabelData, self.CenterTraj = self.LoadTraj()

	def LoadTraj(self):
		print('---------------- Load Trajectories ----------------')
		VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + self.DEP + self.ARR + str(self.Year) + '.csv'
		VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
		LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + self.DEP+'_' + self.ARR+ '_' + str(self.Year) + '.csv', parse_dates=[6])
		CenterTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)

		print('Finished')
		return LabelData, CenterTraj

	def Matching(self):
		# Get spatial query index
		print('---------------- Start horizontal spatial matching----------------')
		self.CenterTraj['levels'] = self.CenterTraj['Alt'].apply(lambda x: proxilvl(x*100, self.Levels))
		self.CenterTraj['QueryIdx'] = 0
		self.CenterTraj['QueryIdx'] = self.CenterTraj['QueryIdx'].astype(int)
		for lvl, gp in self.CenterTraj.groupby('levels'):
		    self.CenterTraj.loc[gp.index, 'QueryIdx'] = self.Grid_KDTree.query(gp[['Lon','Lat']])[1]

		# Prepare for temporal matching
		self.CenterTraj['TimeDelta'] = self.CenterTraj.groupby('FID')['Elap_Time'].transform(lambda x: (x - x.iloc[0]))
		self.CenterTraj['TimeDelta'] = (self.CenterTraj['TimeDelta'] - self.CenterTraj.loc[0,'TimeDelta']).apply(lambda x: x.seconds/3600)
		self.CenterTraj = GetAzimuth(self.CenterTraj)

		print('---------------- Prepare for temporal matching ----------------')
		TimeQuery = []
		self.MemberFID = []
		st = time.time()
		for i in range(self.LabelData.shape[0]):
		    if i % 500 == 0:
		        print(i, time.time() - st)
		    self.MemberFID.append(self.LabelData.loc[i, 'FID'])
		    departureTime = self.LabelData.loc[i, 'Elap_Time']
		    dt = departureTime - BaseTime
		    dt = dt.days * 24. + dt.seconds/3600.
		    TimeQuery.append(self.CenterTraj.TimeDelta.values + dt)
		TimeQuery = np.array(TimeQuery)
		self.MemberFID = np.array(self.MemberFID)
		print('---------------- Start temporal matching----------------')
		TimeDist, TimeQueriedIdx = self.TimeIdxTree.query(TimeQuery.reshape(-1,1))
		TimeQueriedIdx = TimeQueriedIdx.reshape(TimeQuery.shape)

		print('---------------- Start vertical spatial matching----------------')

		    
		level_idx = self.CenterTraj.levels.apply(lambda x: self.level_idx_dict[x]).values
		Temporal_Lvl_Idx = (TimeQueriedIdx * 12 + level_idx.reshape(1,-1)).astype(int)
		Spatial_Lvl_Idx = np.repeat(self.CenterTraj.QueryIdx.values.reshape(1,-1), self.LabelData.shape[0],0)
		print(Spatial_Lvl_Idx.shape)
		print(Temporal_Lvl_Idx.shape)

		print('---------------- Finilize matching and reshaping ----------------')
		matched_u_wind = self.u_wind[Temporal_Lvl_Idx.reshape(-1,1),Spatial_Lvl_Idx.reshape(-1,1)].reshape(TimeQuery.shape)
		matched_v_wind = self.v_wind[Temporal_Lvl_Idx.reshape(-1,1),Spatial_Lvl_Idx.reshape(-1,1)].reshape(TimeQuery.shape)
		matched_headwind = np.multiply(np.sin(self.CenterTraj.azimuth.values * np.pi/180.).reshape(-1,1), matched_u_wind.T) + \
		                    np.multiply(np.cos(self.CenterTraj.azimuth.values * np.pi/180.).reshape(-1,1), matched_v_wind.T)
		matched_wind_dist = np.multiply(self.CenterTraj.DT.values.reshape(-1,1), matched_headwind)


		I_matrix = np.zeros((self.CenterTraj.FID.unique().shape[0], self.CenterTraj.shape[0]))
		I_matrix_mean = np.zeros((self.CenterTraj.FID.unique().shape[0], self.CenterTraj.shape[0]))

		for j in range(I_matrix.shape[0]):
		    try:
		        I_matrix[j, self.CenterTraj.groupby('FID').head(1).index[j]:self.CenterTraj.groupby('FID').head(1).index[j+1]] = 1
		        I_matrix_mean[j, self.CenterTraj.groupby('FID').head(1).index[j]:self.CenterTraj.groupby('FID').head(1).index[j+1]] = 1/np.count_nonzero(I_matrix[j,:])
		    except:
		        I_matrix[j, self.CenterTraj.groupby('FID').head(1).index[j]:] = 1
		        I_matrix_mean[j, self.CenterTraj.groupby('FID').head(1).index[j]:] = 1/np.count_nonzero(I_matrix[j,:])

		self.mean_wind_sp = I_matrix_mean.dot(matched_headwind).T.reshape(-1,1) # m/s
		self.wind_dist_nm = 0.0005399568034555 * I_matrix.dot(matched_wind_dist).T.reshape(-1,1) # nmi
		return self.mean_wind_sp, self.wind_dist_nm, matched_headwind, matched_wind_dist, matched_u_wind, matched_u_wind

	def VisualizeWind(self, AirPressure, Scale = 2500, Time = '02/04/2013 18:00'):
		# create figure and axes instances
	    global BaseTime
	    
	    time_idx = self.TimeIdxTree.query([(parse(Time) - BaseTime).days * 24 + (parse(Time) - BaseTime).seconds])[1]
	    final_idx = int(time_idx * 12 + self.level_idx_dict[AirPressure])
	    u_wind = self.u_wind[final_idx]
	    v_wind = self.v_wind[final_idx]
	    z_wind = np.sqrt(u_wind**2 + v_wind**2)
	    
	    x = self.Grid[:, 0]
	    y = self.Grid[:, 1]
	    
	    xi = np.arange(min(x), max(x) + 0.1, 2.5)
	    yi = np.arange(max(y), min(y) - 0.1, -2.5)
	    
	    
	    fig = plt.figure(figsize=(16,12))
	    plt.title('Wind Speed (m/s), ' + Time + 'Z. Elevation: ' + str(AirPressure) + ' mbar')
	    # create polar stereographic Basemap instance.
	    latlb = 21
	    latub = 50
	    lonlb = -127
	    lonub = -67
	    m = Basemap(projection='merc',llcrnrlat = latlb, urcrnrlat = latub, llcrnrlon = lonlb,urcrnrlon = lonub)
	    m.drawcoastlines(linewidth=1)
	    m.drawstates(linewidth=0.25)
	    m.drawcountries(linewidth=1)
	    
	    x, y = m(x,y)
	    Q = plt.quiver(x,y,u_wind,v_wind, scale = Scale, zorder = 10)

	    X, Y = np.meshgrid(xi, yi)
	    X, Y = m(X, Y)
	    CS = plt.contourf(X, Y, z_wind.reshape(len(yi), len(xi)),cmap=plt.cm.jet, zorder = 0)

	    ax = plt.gca()
	    divider = make_axes_locatable(ax)
	    cax = divider.append_axes("right", size="5%", pad=0.05)
	    cb = plt.colorbar(CS, cax=cax)
	    cb.set_label('m/s')
	    qk = plt.quiverkey(Q, 0.1, 0, 100, '100 m/s', labelpos='W', color = 'r', labelcolor='k', fontproperties = {'size': 14})
	    
	    return z_wind
	def MergeWithMNL(self, Overwrite = False):
		# work directly with MA
		MergingIdx = np.hstack((np.repeat(self.CenterTraj.groupby('FID').head(1).FID.values.reshape(1,-1), self.MemberFID.shape[0], axis = 0).reshape(-1,1), 
								np.repeat(self.MemberFID.reshape(-1,1), self.CenterTraj.FID.unique().shape[0], axis = 0)))
		wind_df = pd.DataFrame(MergingIdx, columns=['NominalFID', 'MemberFID'])
		wind_df['wind_dist'] = self.wind_dist_nm
		wind_df['mean_wind_sp'] = self.mean_wind_sp

		MNL = pd.read_csv(os.getcwd() + '/MNL/MA_Final_MNL_' + self.DEP + self.ARR + '_' + str(self.Year) + '.csv')
		MNL = MNL.merge(wind_df, left_on=['FID_x', 'FID_Member'], right_on=['NominalFID', 'MemberFID'], how = 'left')
		del MNL['NominalFID']
		del MNL['MemberFID']

		if Overwrite:
			MNL.to_csv(os.getcwd() + '/MNL/MA_Final_MNL_' + self.DEP + self.ARR + '_' + str(self.Year) + '.csv', index = False)
		return MNL



# DEP = 'IAH'
# ARR = 'BOS'
# Year = 2013

# print('---------------- Load Trajectories ----------------')
# VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + DEP + ARR + str(Year) + '.csv'
# VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
# LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + DEP+'_' + ARR+ '_' + str(Year) + '.csv', parse_dates=[6])
# CenterTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)

# # Get spatial query index
# print('---------------- Start spatial (horizontal) matching----------------')
# CenterTraj['levels'] = CenterTraj['Alt'].apply(lambda x: proxilvl(x*100, wind_data))
# CenterTraj['QueryIdx'] = 0
# CenterTraj['QueryIdx'] = CenterTraj['QueryIdx'].astype(int)
# for lvl, gp in CenterTraj.groupby('levels'):
#     CenterTraj.loc[gp.index, 'QueryIdx'] = Grid_KDTree.query(gp[['Lon','Lat']])[1]
# # Prepare for temporal matching
# CenterTraj['TimeDelta'] = CenterTraj.groupby('FID')['Elap_Time'].transform(lambda x: (x - x.iloc[0]))
# CenterTraj['TimeDelta'] = (CenterTraj['TimeDelta'] - CenterTraj.loc[0,'TimeDelta']).apply(lambda x: x.seconds/3600)
# CenterTraj = GetAzimuth(CenterTraj)

# print('---------------- Prepare for temporal matching ----------------')
# TimeQuery = []
# MemberFID = []
# st = time.time()
# for i in range(LabelData.shape[0]):
#     if i % 500 == 0:
#         print(i, time.time() - st)
#     MemberFID.append(LabelData.loc[i, 'FID'])
#     departureTime = LabelData.loc[i, 'Elap_Time']
#     dt = departureTime - BaseTime
#     dt = dt.days * 24. + dt.seconds/3600.
#     TimeQuery.append(CenterTraj.TimeDelta.values + dt)
# TimeQuery = np.array(TimeQuery)
# MemberFID = np.array(MemberFID)
# print('---------------- Start temporal matching----------------')
# TimeDist, TimeQueriedIdx = TimeIdxTree.query(TimeQuery.reshape(-1,1))
# TimeQueriedIdx = TimeQueriedIdx.reshape(TimeQuery.shape)


# print('---------------- Start spatial (vertical) matching----------------')
# level_idx_dict = {}
# i = -1
# for key in wind_data.keys():
#     i += 1
#     level_idx_dict[key] = i
    
# level_idx = CenterTraj.levels.apply(lambda x: level_idx_dict[x]).values
# Temporal_Lvl_Idx = TimeQueriedIdx * 12 + level_idx.reshape(1,-1)
# Spatial_Lvl_Idx = np.repeat(CenterTraj.QueryIdx.values.reshape(1,-1), LabelData.shape[0],0)
# print(Spatial_Lvl_Idx.shape)
# print(Temporal_Lvl_Idx.shape)

# print('---------------- Finilize matching and reshaping ----------------')
# matched_u_wind = u_wind[Temporal_Lvl_Idx.reshape(-1,1),Spatial_Lvl_Idx.reshape(-1,1)].reshape(TimeQuery.shape)
# matched_v_wind = v_wind[Temporal_Lvl_Idx.reshape(-1,1),Spatial_Lvl_Idx.reshape(-1,1)].reshape(TimeQuery.shape)
# matched_headwind = np.multiply(np.sin(CenterTraj.azimuth.values * np.pi/180.).reshape(-1,1), matched_u_wind.T) + \
#                     np.multiply(np.cos(CenterTraj.azimuth.values * np.pi/180.).reshape(-1,1), matched_v_wind.T)
# matched_wind_dist = np.multiply(CenterTraj.DT.values.reshape(-1,1), matched_headwind)

# I_matrix = np.zeros((CenterTraj.FID.unique().shape[0], CenterTraj.shape[0]))
# I_matrix_mean = np.zeros((CenterTraj.FID.unique().shape[0], CenterTraj.shape[0]))

# for j in range(I_matrix.shape[0]):
#     try:
#         I_matrix[j, CenterTraj.groupby('FID').head(1).index[j]:CenterTraj.groupby('FID').head(1).index[j+1]] = 1
#         I_matrix_mean[j, CenterTraj.groupby('FID').head(1).index[j]:CenterTraj.groupby('FID').head(1).index[j+1]] = 1/np.count_nonzero(I_matrix[j,:])
#     except:
#         I_matrix[j, CenterTraj.groupby('FID').head(1).index[j]:] = 1
#         I_matrix_mean[j, CenterTraj.groupby('FID').head(1).index[j]:] = 1/np.count_nonzero(I_matrix[j,:])

# mean_wind_sp = I_matrix_mean.dot(matched_headwind).T.reshape(-1,1)
# wind_dist_nm = 0.0005399568034555 * I_matrix.dot(matched_wind_dist).T.reshape(-1,1)