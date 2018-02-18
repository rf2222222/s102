# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 06:12:03 2015

@author: hidemi
"""

import math
import numpy as np
import pandas as pd
import random
import pickle
import matplotlib
from datetime import datetime as dt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from IPython.core.display import display_png
#import pydot

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor,\
                        ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import scale, robust_scale, minmax_scale
from suztoolz.transform import zigzag as zg

#import warnings
#warnings.filterwarnings('error')

def calcDPS(signal_type, sst, PRT, windowLength, **kwargs):
    #threshold=kwargs.get('threshold',-np.inf)
    trade=kwargs.get('trade','both')
    asset = kwargs.get('asset','FX')
    startDate= kwargs.get('startDate', sst.index[-1])
    endDate = kwargs.get('endDate',sst.index[-1])
    verbose = kwargs.get('verbose',True)
    
    def IBcommission(tradeAmount, asset):
        commission = 2.0
        if asset == 'FX':
            return max(2.0, tradeAmount*2e-5)
        else:
            return commission
            
    def calcEquityLast(i, sst,safef):
        #safef is calculated in dps, should be nan in sst. 
        #print safef
        if 'dpsSafef' in sst:
            if (sst.iloc[i-1].signals*sst.iloc[i-1].dpsSafef < 0):
                equityBeLongAndShortSignals = (1+sst.iloc[i-1].dpsSafef*-sst.iloc[i-1].gainAhead)*sst.iloc[i-1].dpsNetEquity
            elif (sst.iloc[i-1].signals*sst.iloc[i-1].dpsSafef > 0):
                equityBeLongAndShortSignals= (1+sst.iloc[i-1].dpsSafef*sst.iloc[i-1].gainAhead)*sst.iloc[i-1].dpsNetEquity
            else:
                equityBeLongAndShortSignals = sst.iloc[i-1].dpsNetEquity
                
            positionChg = abs(sst.signals[i]*safef-sst.signals[i-1]*sst.dpsSafef[i-1])
            #print positionChg
            if positionChg !=0:
                commission = IBcommission(positionChg*equityBeLongAndShortSignals, asset)
            else:
                commission = 0.0
            lastEquity = round(equityBeLongAndShortSignals-commission,2)
            netPNL = round(lastEquity - sst.iloc[i-1].dpsNetEquity,2)
            return netPNL, commission, lastEquity
            
        else:
            return 0.0, 0.0, sst.iloc[i-1].netEquity

        
    #updated: -1 short, 0 flat, 1 long
    if windowLength <=1:
        if verbose:
            print 'windowLength needs to be >1 adjusting to 1.5'
        windowLength = 1.5
        
    windowLength = float(windowLength)
    initialEquity = PRT['initial_equity']
    ddTolerance = PRT['DD95_limit']
    tailRiskPct = PRT['tailRiskPct']
    forecastHorizon = PRT['horizon']
    maxLeverage = PRT['maxSafef']
    minSafef= PRT['minSafef']
    threshold=PRT['CAR25_threshold']
    
    nCurves = 50
    accuracy_tolerance = 0.005
    updateInterval = 1
    #normalize personal risk tolerance by windowLength for safef
    multiplier = ddTolerance/math.sqrt(forecastHorizon) #assuming dd increases with sqrt of time
    forecastHorizon = windowLength/2.3 #need a bit more than 2x trades of the fcst horizon
    ddTolerance = math.sqrt(forecastHorizon)* multiplier #adjusted dd tolerance for the forecast
    #startDate = sst.index[-1]
    
    if startDate==endDate:
        #returns last index only
        #years_in_forecast =(sst.index[-1]-sst.index[-2]).total_seconds()/3600.0/24.0/365.0
        #to rank worst case better.
        years_in_forecast =(sst.index[-1]-sst.index[-2]).total_seconds()/3600.0
    else:
        years_in_forecast = (endDate-startDate).total_seconds()/3600.0/24.0/365.0
        
    iStart = sst.index.get_loc(startDate)
    iEnd = sst.index.get_loc(endDate)    
    #years_in_forecast = forecastHorizon / 252.0
    dpsRunName = trade + '_'+signal_type + ' DPS wl%.1f maxL%i dd95_%.3f thres_%.1f'\
                            % (windowLength,maxLeverage,ddTolerance,threshold)
    if verbose:
        print '\n', dpsRunName, 'from', startDate, 'to', endDate
    #  Work with the index rather than the date

    
    safef_ser = pd.Series(index =  range(iStart,iEnd+1), name='dpsSafef')
    CAR25_ser = pd.Series(index =  range(iStart,iEnd+1), name='CAR25')
    dd95_ser = pd.Series(index =  range(iStart,iEnd+1), name='dd95')
    ddTol_ser = pd.Series(data=ddTolerance, index =  range(iStart,iEnd+1), name='ddTol')
    dpsPNL_ser = pd.Series(index =  range(iStart,iEnd+1), name='dpsNetPNL')
    dpsEquity_ser = pd.Series(index =  range(iStart,iEnd+1), name='dpsNetEquity')
    dpsCommission_ser = pd.Series(index =  range(iStart,iEnd+1), name='dpsCommission')
    dpsRunName_ser = pd.Series(index =  range(iStart,iEnd+1), name='dpsRunName')

    for i in range(iStart, iEnd+1, updateInterval):
        #print '.',
        #print '\n', i,sst.index[i], sst.signals[i], sst.gainAhead[i]
    
    #  Initialize variables
        curves = np.zeros(nCurves)
        numberDraws = np.zeros(nCurves)
        TWR = np.zeros(nCurves)
        maxDD = np.zeros(nCurves)
        
        safef = 1.00
        dd95 = 2 * ddTolerance
        done = False
        
        while not done:
            #  Generate nCurve equity curves

            for nc in range(nCurves):
                equity = initialEquity
                maxEquity = equity
                drawdown = 0
                maxDrawdown = 0
                horizonSoFar = 0
                nd = 0
                lastPosition = 0
                while (horizonSoFar < forecastHorizon):
                #get signals from starting from yesterday to windowlength
                    j = float(np.random.randint(1,windowLength+1))
                    if sst.index[i] < sst.index[i-int(j)]:
                        raise Exception('i '+str(sst.index[i])+ ' < i-j ' +str(sst.index[i-int(j)])+', '+\
                                            'choose a shorter window length or add more data') 
                    nd = nd + 1
                    weightJ = 1.00 - j/windowLength #int div int = int
                    horizonSoFar = horizonSoFar + weightJ
                    signalsJ = sst.signals[i-int(j)] #j as int for indexing
                    # trade is long, then long only, short then short only, both otherwise, 0 is flat
                    if signalsJ > 0:
                        if trade == 'short':
                            #print 'beFlat'
                            tradeJ = 0.0
                        else:  # trade == 'long' or trade == 'both'
                            #print 'beLong'
                            tradeJ = sst.gainAhead[i-int(j)] * weightJ                        
                    elif signalsJ <0:
                        if trade == 'long': 
                            #print 'beFlat'
                            tradeJ = 0.0
                        else:# trade == 'short' or trade == 'both'
                            #print 'beShort'
                            tradeJ = -sst.gainAhead[i-int(j)] * weightJ                
                    else: #signalJ ==0
                        #print 'beFlat'
                        tradeJ = 0.0
                        
                    #commissions
                    if signalsJ == lastPosition:
                        commission=0
                    else:
                        commission = IBcommission(safef*equity, asset)                            

                    thisTrade = safef * tradeJ * equity    
                    equity = equity + thisTrade -commission
                    maxEquity = max(equity,maxEquity)
                    drawdown = (maxEquity-equity)/maxEquity
                    maxDrawdown = max(drawdown,maxDrawdown)
                    #print "thistrade, equity, signalsj, sst.gainAhead[i-j], tradej, safef"
                    #print thisTrade, equity, signalsJ, sst.gainAhead[i-j], tradeJ, safef
                    #print "maxDD, ndraws, horizonsofar, fcsthor:",maxDrawdown, nd, horizonSoFar, forecastHorizon
                #print nc, "\n\nCURVE DONE equity, maxDD, ndraws:", equity, maxDrawdown, nd      
                TWR[nc] = equity
                maxDD[nc] = maxDrawdown
                numberDraws[nc] = nd
        
            #  Find the drawdown at the tailLimit-th percentile        
            dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
            #print 'maxdd', maxDD
            #print "  DD %i: %.3f " % (tailRiskPct, dd95)
            safef = safef * ddTolerance / dd95
            #print safef,
            TWR25 = stats.scoreatpercentile(TWR,25)        
            CAR25 = 100*(((TWR25/initialEquity) ** 
                      (1.0/years_in_forecast))-1.0)
            #print 'twr, car25,dd95, ddtol', TWR25, CAR25,dd95, ddTolerance
            #print 'dd95-ddtol', abs(ddTolerance-dd95)
    
            if safef > maxLeverage:
                #safef = maxLeverage
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "safef > maxLeverage" 
                done = True
            elif (abs(ddTolerance - dd95) < accuracy_tolerance):
                #print 'dd95-ddtol', abs(ddTolerance-dd95), 'accuracy_tolerance', accuracy_tolerance
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "Close enough" 
                done = True
                
        #while done
        if CAR25 > threshold:
            #print safef, CAR25, dd95
            safef_ser[i] = maxLeverage
            CAR25_ser[i] = CAR25
            dd95_ser[i] = dd95
            (dpsPNL_ser[i], dpsCommission_ser[i], dpsEquity_ser[i])= calcEquityLast(i, sst, maxLeverage)
            dpsRunName_ser[i] = dpsRunName

        else:
            #minSafef
            safef_ser[i] = minSafef
            CAR25_ser[i] = CAR25
            dd95_ser[i] = threshold
            (dpsPNL_ser[i], dpsCommission_ser[i], dpsEquity_ser[i])=calcEquityLast(i, sst, minSafef)
            dpsRunName_ser[i] = dpsRunName
    
    #print "Done!"
    if 'dpsSafef' in sst:
        sst_save = sst.drop(['CAR25','dd95','ddTol','dpsNetPNL','dpsNetEquity','dpsRunName',\
                                        'dpsSafef','dpsCommission'], axis=1).reset_index()
    else:
        sst_save = sst.reset_index()
        
    sst_save = pd.concat([sst_save.ix[iStart:iEnd], dpsPNL_ser, dpsEquity_ser, safef_ser, CAR25_ser,\
                                    dd95_ser, ddTol_ser, dpsRunName_ser,dpsCommission_ser], axis=1)
    sst_save= sst_save.set_index(pd.DatetimeIndex(sst_save['dates'])).drop(['dates'], axis=1)
    sst_save.index.name = 'dates'
    #print sst_save.tail()
    #DPS[dpsRun] = sst_save
    return sst_save
        
