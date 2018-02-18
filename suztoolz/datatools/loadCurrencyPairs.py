# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import math
import talib as ta
import pandas as pd
from suztoolz.transform import zigzag as zg
import arch
from os import listdir
from os.path import isfile, join

from datetime import datetime
import matplotlib.pyplot as plt
#from pandas.io.dataSet import DataReader
import random
from scipy import stats
from scipy.stats import kurtosis, skew
import time
from suztoolz.transform import perturb_data
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio

from suztoolz.loops import CAR25, CAR25_prospector, maxCAR25
from suztoolz.display import init_report, update_report_prospector,\
                            display_CAR25, compareEquity_vf, getToxCDF, adf_test,\
                            describeDistribution
from sklearn.grid_search import ParameterGrid
import re
import copy
import string
from os import listdir
from os.path import isfile, join
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import datetime
from datetime import datetime as dt
from pandas.core import datetools
import time
from suztoolz.transform import RSI, ROC, zScore, softmax, DPO, numberZeros,\
                        gainAhead, ATR, priceChange, garch, autocorrel, kaufman_efficiency,\
                        volumeSpike, softmax_score, create_indicators, ratio, perturb_data,\
                        roofingFilter, getCycleTime, saveParams

from suztoolz.loops import calcDPS2, calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.display import displayRankedCharts,offlineMode
from sklearn.preprocessing import scale, robust_scale, minmax_scale
import logging
import os
from pytz import timezone
from dateutil.parser import parse

    
def loadCurrencyPairs(currencyPairs, dataPath, barSizeSetting, maxlb, ticker,\
                                        signalPath, version, version_, maxReadLines,\
                                        **kwargs):
    perturbData =  kwargs.get('perturbData',False)
    perturbDataPct=kwargs.get('perturbDataPct',0.0002)
    verbose=kwargs.get('verbose',False)
    addAuxPairs=kwargs.get('addAuxPairs',False)
        
    currencyPairsDict = {}
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]

    if addAuxPairs:    
        for pair in currencyPairs:    
            if barSizeSetting+'_'+pair+'.csv' in files and (ticker[0:3] in pair or ticker[3:6] in pair):
                data = pd.read_csv(dataPath+barSizeSetting+'_'+pair+'.csv', index_col=0)
                if data.shape[0] < maxlb:
                    if pair == ticker:
                        message =  'Not enough data to create indicators: #rows\
                            is less than max lookback of '+str(maxlb)
                        offlineMode(ticker, message, signalPath, version, version_)
                    print( 'Skipping aux pair '+pair+'. Not enough data to create indicators: '+\
                                        str(data.shape[0])+' rows is less than max lookback of '+str(maxlb))
                elif data.shape[0] >= maxReadLines:
                    currencyPairsDict[pair] = data[-maxReadLines:]
                else:
                    currencyPairsDict[pair] = data
                    
        nrows = currencyPairsDict[ticker].shape[0]
        
        #print 'removing dupes..'
        currencyPairsDict2 = copy.deepcopy(currencyPairsDict)
        for pair in currencyPairsDict2:
            data = currencyPairsDict2[pair].copy(deep=True)
            #perturb dataSet
            if perturbData:
                if verbose:
                    print 'perturbing OHLC and dropping dupes',
                data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
                data['High']= perturb_data(data['High'].values,perturbDataPct)
                data['Low']= perturb_data(data['Low'].values,perturbDataPct)
                data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
                
            if verbose:
                print 'Dropping dupes:', pair, currencyPairsDict2[pair].shape,'to',
            currencyPairsDict2[pair] = data.drop_duplicates()
            if verbose:
                print pair, currencyPairsDict2[pair].shape

        dataSet=currencyPairsDict2[ticker].copy(deep=True)
        lastIndex = dataSet.index[-1]       
        
        if verbose:            
            for pair in currencyPairsDict2:
                if dataSet.shape[0] != currencyPairsDict2[pair].shape[0]:
                    print 'Warning:',ticker, pair, 'row mismatch. Some Data may be lost.'
                    
        #align the index, at the end of the loop, dataSet should have an index that all pairs contain
        for pair in currencyPairsDict2:
            missingData=currencyPairsDict2[ticker].index.sym_diff(currencyPairsDict2[pair].index)
            print 'Missing data:',ticker,pair, missingData
            
            #forward fill missing data
            if len(missingData)>0:
                print 'Forward filling missing data..'
                #print currencyPairsDict2[pair].ix[missingData]
                currencyPairsDict2[pair]=currencyPairsDict2[pair].ix[currencyPairsDict[ticker].index].fillna(method='ffill')
                currencyPairsDict2[ticker]=currencyPairsDict2[ticker].ix[currencyPairsDict[pair].index].fillna(method='ffill')
                #print currencyPairsDict2[pair].ix[missingData]
                #first index not forwardfillable
                currencyPairsDict2[pair]=currencyPairsDict2[pair].dropna()        
                currencyPairsDict2[ticker]=currencyPairsDict2[ticker].dropna()
                
            if pair != ticker:
                intersect =np.intersect1d(currencyPairsDict2[pair].index, dataSet.index)
                dataSet = dataSet.ix[intersect]
        #align the other pairs.
        for pair in currencyPairsDict2:
            #print 'Reindex:',pair, currencyPairsDict2[pair].shape,
            currencyPairsDict2[pair] = currencyPairsDict2[pair].ix[dataSet.index]
            currencyPairsDict2[pair] = currencyPairsDict2[pair].fillna(method='ffill')
            #print 'to', currencyPairsDict2[pair].shape
                
        dataSet=currencyPairsDict2[ticker].copy(deep=True)
        print nrows-dataSet.shape[0], 'rows lost for', ticker   
        print currencyPairsDict[ticker].index.sym_diff(dataSet.index)
        
        currencyPairsDict3 ={}
        for pair in currencyPairsDict2:
            if pair !=ticker:
                closes = pd.concat([dataSet.Close, currencyPairsDict2[pair].Close],\
                                        axis=1, join='inner')
                highs = pd.concat([dataSet.High, currencyPairsDict2[pair].High],\
                                axis=1, join='inner')
                lows = pd.concat([dataSet.Low, currencyPairsDict2[pair].Low],\
                                        axis=1, join='inner')
                #check if there is enough data to create indicators
                if closes.shape[0] < maxlb:
                    message = 'Not enough data to create indicators: intersect of '\
                        +ticker+' and '+pair +' of '+str(closes.shape[0])+\
                        ' is less than max lookback of '+str(maxlb)
                    offlineMode(ticker, message, signalPath, version, version_)
                else:
                    currencyPairsDict3[pair] = {'closes':closes,'highs':highs,'lows':lows}
                    
        return dataSet, currencyPairsDict3
    else:
        if barSizeSetting+'_'+ticker+'.csv' in files:
            data = pd.read_csv(dataPath+barSizeSetting+'_'+ticker+'.csv', index_col=0)
            if data.shape[0] < maxlb:
                message =  'Not enough data to create indicators: #rows\
                    is less than max lookback of '+str(maxlb)
                offlineMode(ticker, message, signalPath, version, version_)
            elif data.shape[0] >= maxReadLines:
                currencyPairsDict[ticker] = data[-maxReadLines:]
            else:
                currencyPairsDict[ticker] = data

        nrows = currencyPairsDict[ticker].shape[0]
        
        #print 'removing dupes..'
        currencyPairsDict2 = copy.deepcopy(currencyPairsDict)
        data = currencyPairsDict2[ticker].copy(deep=True)
        #perturb dataSet
        if perturbData:
            if verbose:
                print 'perturbing OHLC and dropping dupes',
            data['Open'] = perturb_data(data['Open'].values,perturbDataPct)
            data['High']= perturb_data(data['High'].values,perturbDataPct)
            data['Low']= perturb_data(data['Low'].values,perturbDataPct)
            data['Close'] = perturb_data(data['Close'].values,perturbDataPct)
            
        if verbose:
            print ticker, currencyPairsDict2[ticker].shape,'to',
        currencyPairsDict2[ticker] = data.drop_duplicates()
        if verbose:
            print ticker, currencyPairsDict2[ticker].shape
        dataSet=currencyPairsDict2[ticker].copy(deep=True)
        return dataSet, currencyPairsDict2