# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:30:05 2017

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
import pandas as pd
import os
import collections
import numpy as np
import statsmodels.api as sm

def GetSeason(x):
    if x >= 12 or x <= 2:
        return 0 # Winter
    elif x <= 5:
        return 1 # Spring
    elif x <= 8:
        return 2 # Summer
    else:
        return 3 # Fall

class PreProcessMNL:
    def __init__(self, DEP, ARR, Year):
        self.DEP = DEP
        self.ARR = ARR
        self.Year = Year

    def GetMNL_Data(self):

        # Calculate Great Circle Distance of Nominal Routes
        # Calculate Standard Wind Distance
        # Use Upper Bound for Outlier Group
        # Return Processed MNL Data
        MNL_DATA = pd.read_csv(os.getcwd() + '/MNL/MA_Final_MNL_' + self.DEP + self.ARR + '_' + str(self.Year) + '.csv', header = 0)
        
        VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + self.DEP + self.ARR + str(self.Year) + '.csv'
        VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
        LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + self.DEP+'_' + self.ARR+ '_' + str(self.Year) + '.csv', parse_dates=[6])
        CenterTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)
        # Calculate GC Distance
        GC_Dist = CenterTraj.groupby('FID').Dist.sum().reset_index().set_index('FID')
        GC_Dist.columns = ['GC_Dist']
        MNL_DATA = MNL_DATA.merge(GC_Dist, left_on='FID_x', how = 'left', right_index= True)

        # Process Outlier and Standard Wind Distance
        MissingFID = MNL_DATA.groupby('FID_Member')['Wind_Dist'].sum().reset_index()
        MissingFID = MissingFID[MissingFID.Wind_Dist.isnull()]['FID_Member'].values
        modelset = MNL_DATA[~MNL_DATA.FID_Member.isin(MissingFID)].reset_index(drop = 1)

# Large wind dataset
#         modelset.loc[modelset.FID_x == 99999999, ['Wind_Dist','MeanWindSpeed','GC_Dist']] = modelset.groupby('FID_Member')['Wind_Dist',\
#                                                             'MeanWindSpeed','GC_Dist'].transform('max')
#         # Wind Distance: 1000 nmi
#         modelset.loc[:,'Std_Wind_Dist'] = (modelset['GC_Dist'] - modelset['Wind_Dist'])/1000
        
# #         modelset.loc[modelset.FID_x == 99999999, ['Std_Wind_Dist']] = modelset.groupby('FID_Member')['Std_Wind_Dist'].transform('min')
#         modelset.loc[modelset.Wind_Dist.isnull(), ['Wind_Dist','MeanWindSpeed',\
#                     'Std_Wind_Dist','GC_Dist']] = modelset.groupby('FID_Member')['Wind_Dist',\
#                                                             'MeanWindSpeed','Std_Wind_Dist','GC_Dist'].transform('mean')

# Small wind dataset
        modelset.loc[modelset.FID_x == 99999999, ['wind_dist','mean_wind_sp','GC_Dist']] = modelset.groupby('FID_Member')['wind_dist',\
                                                            'mean_wind_sp','GC_Dist'].transform('max')
        # Wind Distance: 1000 nmi
        modelset.loc[:,'Std_Wind_Dist'] = (modelset['GC_Dist'] - modelset['wind_dist'])/1000
        modelset.loc[modelset.wind_dist.isnull(), ['wind_dist','mean_wind_sp',\
                    'Std_Wind_Dist','GC_Dist']] = modelset.groupby('FID_Member')['wind_dist',\
                                                            'mean_wind_sp','Std_Wind_Dist','GC_Dist'].transform('mean')

        modelset.loc[:,'MIT_Str_sum'] = modelset['MIT_Str_sum'].fillna(0)
        modelset.loc[:,'MIT_Str_mean'] = modelset['MIT_Str_mean'].fillna(0)
        modelset.loc[:,'MIT_Str_max'] = modelset['MIT_Str_max'].fillna(0)
        modelset.loc[:,'MIT_Str_sum'] = modelset['MIT_Str_sum']/1000
        modelset.loc[:,'MIT_Str_mean'] = modelset['MIT_Str_mean']/1000
        modelset.loc[:,'MIT_Str_max'] = modelset['MIT_Str_max']/1000
        modelset.loc[:,'AvgTimeRed'] = modelset['AvgTimeRed']/60
        modelset.loc[:,'AvgTimeYellow'] = modelset['AvgTimeYellow']/60
        modelset['TotalAlert'] = modelset['NumRed'] + modelset['NumYellow']
        modelset['BusyHour'] = modelset.LocalHour.apply(lambda x: (x>=10) & (x<=20)).astype(int)
#         modelset['MIT_Impact'] = modelset['MITVAL'] * modelset['DURATION_HRS_TOT']
#         print(modelset.FID_Member.unique().shape[0])
        return modelset
    
    def GetLR_Data(self, wind_metric = 'Std_Wind_Dist', outlier = False, OneMetric = False):
        Data_ORI = self.GetMNL_Data()
        Data_ORI['OD'] = self.DEP + '_' + self.ARR
        Data_ORI['OD_Clust'] = self.DEP + '_' + self.ARR + '_' + Data_ORI.Alt_id.map(str)
        Data_ORI.loc[Data_ORI.FID_x == 99999999,'OD_Clust'] = self.DEP + '_' + self.ARR + '_-1'
        
        ASC_ColumnIdx = range(2,18)
        # Weather, Wind, MIT
        ASC_ColumnIdx.extend([49,50,51,52,53,54,55,56,57,58,59,60])
        # Ineff, clustID, localmonth, local hour, seaon, busyhour
        SD_ColumnIdx = [22,21,28,29,44,63]
        
        if OneMetric:
            LR_Data = Data_ORI[Data_ORI.CHOICE == 1].reset_index(drop = 1)
            LR_Data = LR_Data[LR_Data.columns[[1] + SD_ColumnIdx + ASC_ColumnIdx]]
        else:        
            if outlier:
                pass
            else:
                Data = Data_ORI[Data_ORI.FID_x != 99999999].reset_index(drop = True)
                
            wt = Data[Data.CHOICE == 1].groupby('FID_x').FID_Member.count()/Data[Data.CHOICE == 1].FID_Member.unique().shape[0]
            weight = lambda x: wt.ix[x]
    
            Data['wt'] = Data.groupby('FID_x').FID_x.transform(weight)
    
            Data[Data.columns[ASC_ColumnIdx]] = Data[Data.columns[ASC_ColumnIdx]].apply(lambda x: x * Data.wt)
    
            Keys = collections.OrderedDict()
            Keys['Efficiency'] = 'mean'
    
            for key in Data.columns[SD_ColumnIdx]:
                Keys[key] = 'mean'
            for key in Data.columns[ASC_ColumnIdx]:
                Keys[key] = 'sum'    
            LR_Data = Data.groupby('FID_Member').agg(Keys).reset_index(drop = False)
            WD_Corr = Data[Data.FID_x != 99999999].groupby('FID_Member')[['GC_Dist',\
                   wind_metric]].cov().ix[0::2,wind_metric].reset_index()[['FID_Member',wind_metric]]
            WD_Corr.columns = ['FID_Member','wind_metric']
            LR_Data = LR_Data.merge(WD_Corr, left_on='FID_Member', right_on='FID_Member')
            
        LR_Data['OD'] = self.DEP + '_' + self.ARR
        LR_Data['OD_Clust'] = self.DEP + '_' + self.ARR + '_' + LR_Data.ClustID.map(str)
#        LR_Data['Season'] = LR_Data.LocalMonth.apply(lambda x: GetSeason(x))
#        LR_Data['BusyHour'] = LR_Data.LocalHour.apply(lambda x: (x>=10) & (x<=20)).astype(int)
        return LR_Data, Data_ORI
    
class OLS_Model:
    def __init__(self,Dep, Arr, Year, OneMetric = False):
        PostMNL = PreProcessMNL(Dep, Arr, Year)
        self.LR_Data1, self.MNL_Data = PostMNL.GetLR_Data(wind_metric = 'Wind_Dist', OneMetric=OneMetric)
        self.LR_Data1.Efficiency = self.LR_Data1.Efficiency * 100

    def fitModel(self,formula):
        self.mod = sm.OLS.from_formula(formula, data=self.LR_Data1).fit()
        return self.mod