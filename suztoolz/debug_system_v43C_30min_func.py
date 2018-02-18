# -*- coding: utf-8 -*-
"""
v4.0
premise:
The premise for this system is that there is an insample period in the recent history
that is in sync with the future market out of sample period.  This insample period can be 
using various methods. More than one in-sample period can be used for prediction.

The system starts with a high/low volatility prediction. If the volatility is high,
then the system makes predictions with a bias toward a new trend. If the volatility is low,
the system makes predictions with a bias toward a recurring cycle. 

Additional bias can be added as a parameter. 

Major Parameters to be optemised:
bar size
support/resistance lookback
adfPvalue
AddAuxPairs & nfeatures


Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import sys
import numpy as np
import math
import talib as ta
import pandas as pd
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
from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
#classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

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

from suztoolz.loops import calcEquity2, createBenchmark, createYearlyStats, findBestDPS
from suztoolz.datatools.loadCurrencyPairs import loadCurrencyPairs
from suztoolz.datatools.acPeriodogram import acPeriodogram
from suztoolz.datatools.zigzag import zigzag as zg
from suztoolz.datatools.mrClassifier import mrClassifier
from suztoolz.position_sizing.calcDPS import calcDPS
from sklearn.preprocessing import scale, robust_scale, minmax_scale
from sklearn.pipeline import Pipeline
import logging
import os
from pytz import timezone
from dateutil.parser import parse

#start_time = time.time()
def average(l):
    return sum(l)/len(l)
    
def IBcommission(tradeAmount, asset):
    commission = 2.0
    if asset == 'FX':
        return max(2.0, tradeAmount*2e-5)
    else:
        return commission

def initSST(close, version_, st, initialEquity):
    savedShadowTrades = pd.DataFrame(index=close.index)
    savedShadowTrades.index.name = 'dates'
    #set to 0 for all signals to start at the same time. 
    savedShadowTrades['signals']=0
    savedShadowTrades['signalType']=st
    savedShadowTrades[version_+'_system']=st
    savedShadowTrades['gainAhead']=close.pct_change().shift(-1).fillna(0)
    savedShadowTrades['netPNL']=0
    savedShadowTrades['nodpsComm']=0
    savedShadowTrades['nodpsSafef']=0
    savedShadowTrades['netEquity']=initialEquity
    
    return savedShadowTrades
        
def calcEquityLast(i, sst):
    #equityBeLongAndShortSignals = np.zeros(nrows)
    #equityBeLongAndShortSignals[0] = sst.equity[-1]
    #for i in range(1,nrows):
    if (sst.signals.iloc[i-1] < 0):
        equityBeLongAndShortSignals = (1+-sst.gainAhead.iloc[i-1]*sst.nodpsSafef.iloc[i-1])*sst.netEquity[i-1]
    elif (sst.signals.iloc[i-1] > 0):
        equityBeLongAndShortSignals= (1+sst.gainAhead.iloc[i-1]*sst.nodpsSafef.iloc[i-1])*sst.netEquity[i-1]
    else:
        equityBeLongAndShortSignals = sst.netEquity[i-1]
    
    positionChg = abs(sst.signals[i]*sst.nodpsSafef.iloc[i]-sst.signals[i-1]*sst.nodpsSafef.iloc[i-1])
    if positionChg !=0:
        commission = IBcommission(positionChg*equityBeLongAndShortSignals, asset)
    else:
        commission = 0.0
        
    lastEquity = round(equityBeLongAndShortSignals-commission,2)
    netPNL = round(lastEquity - sst.netEquity[i-1], 2)       
    
    return netPNL, commission, lastEquity
    
def reCalcEquity(sst, metric):
    #no dps
    nrows=sst.shape[0]
    index = sst.index
    noDpsEquity = np.zeros(nrows)
    netPNL = np.zeros(nrows)
    commission = np.zeros(nrows)
    noDpsEquity[0] = sst.netEquity[0]
    if metric != 'CAR25':
        sst['RS_'+metric] = sst[metric].values
        
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] *sst.nodpsSafef.iloc[i-1]< 0):
            #print sst.signals.iloc[i-1] , sst.nodpsSafef.iloc[i-1], sst.gainAhead.iloc[i-1]
            noDpsEquity[i] = (1+-sst.gainAhead.iloc[i-1]*sst.nodpsSafef.iloc[i-1])*noDpsEquity[i-1]
        elif (sst.signals.iloc[i-1]*sst.nodpsSafef.iloc[i-1] > 0):
            noDpsEquity[i]= (1+sst.gainAhead.iloc[i-1]*sst.nodpsSafef.iloc[i-1])*noDpsEquity[i-1]
        else:
            noDpsEquity[i] = sst.netEquity[i-1]
        
        positionChg = abs(sst.signals[i]*sst.nodpsSafef.iloc[i]-sst.signals[i-1]*sst.nodpsSafef.iloc[i-1])
        
        if positionChg !=0:
            commission[i] = IBcommission(positionChg*noDpsEquity[i], asset)
        else:
            commission[i] = 0.0
        #print sst.signals[i-1], sst.signals[i], positionChg, commission[i]
        noDpsEquity[i] = round(noDpsEquity[i]-commission[i],2)
        netPNL[i] = round(noDpsEquity[i] - noDpsEquity[i-1], 2)
        #print i, noDpsEquity[i]
    sst.set_value(index, 'netEquity', noDpsEquity)
    sst.set_value(index,'netPNL',netPNL)
    sst.set_value(index,'nodpsComm',commission)
    
    #dps
    dpsNetEquity = np.zeros(nrows)
    dpsNetPNL = np.zeros(nrows)
    dpsCommission = np.zeros(nrows)
    dpsNetEquity[0] = sst.dpsNetEquity[0]
    
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] *sst.dpsSafef.iloc[i-1]< 0):
            dpsNetEquity[i] = (1+-sst.gainAhead.iloc[i-1]*sst.dpsSafef.iloc[i-1])*dpsNetEquity[i-1]
        elif (sst.signals.iloc[i-1]*sst.dpsSafef.iloc[i-1] > 0):
            dpsNetEquity[i]= (1+sst.gainAhead.iloc[i-1]*sst.dpsSafef.iloc[i-1])*dpsNetEquity[i-1]
        else:
            dpsNetEquity[i] = sst.netEquity[i-1]
        
        positionChg = abs(sst.signals[i]*sst.dpsSafef.iloc[i]-sst.signals[i-1]*sst.dpsSafef.iloc[i-1])
        if positionChg !=0:
            dpsCommission[i] = IBcommission(positionChg*dpsNetEquity[i], asset)
        else:
            dpsCommission[i] = 0.0
            
        dpsNetEquity[i] = round(dpsNetEquity[i]-dpsCommission[i],2)
        dpsNetPNL[i] = round(dpsNetEquity[i] - dpsNetEquity[i-1], 2)
        
    sst.set_value(index,'dpsNetEquity', dpsNetEquity)
    sst.set_value(index,'dpsNetPNL',dpsNetPNL)
    sst.set_value(index,'dpsCommission', dpsCommission)
    
    return sst
    
def createSignalFile(version, version_, ticker, barSizeSetting, signalPath, sst, start_time, dataSet):
    print version_, 'Saving',ticker, 'Signals..'      
    timenow, lastBartime, cycleTime = getCycleTime(start_time, dataSet)
    files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
    #new version_ file
    if version_+'_'+ ticker+'_'+barSizeSetting+ '.csv' not in files:
        #signalFile = sst.iloc[-2:]
        addLine = sst.iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = sst.iloc[-1].name
        signalFile = sst.iloc[-2:-1].append(addLine)
        signalFile.index.name = 'dates'
        filename = signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv'
        print 'Saving', filename        
        signalFile.to_csv(filename, index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version_+'_'+ ticker+'_'+barSizeSetting+ '.csv', index_col=['dates'])
        addLine = sst.iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = sst.iloc[-1].name
        signalFile = signalFile.append(addLine)
        filename = signalPath + version_+'_'+ ticker+'_'+barSizeSetting+ '.csv'
        print 'Saving', filename        
        signalFile.to_csv(filename, index=True)
        
    #old version_ file
    if version+'_'+ ticker+ '.csv' not in files:
        #signalFile = sst.iloc[-2:]
        addLine = sst.iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = sst.iloc[-1].name
        signalFile = sst.iloc[-2:-1].append(addLine)
        signalFile.index.name = 'dates'
        filename=signalPath + version+'_'+ ticker+ '.csv'
        print 'Saving', filename
        signalFile.to_csv(filename, index=True)
    else:        
        signalFile=pd.read_csv(signalPath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
        addLine = sst.iloc[-1]
        addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
        addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
        addLine.name = sst.iloc[-1].name
        signalFile = signalFile.append(addLine)
        filename=signalPath + version+'_'+ ticker+ '.csv'
        print 'Saving', filename       
        signalFile.to_csv(filename, index=True)

def runv4(runData):
    start_time = time.time()
    global showIndicators
    global showCharts
    globals().update(runData)
    chartSavePath = './data/results/'+version+'_'+ticker
    #print showCharts, showFinalChartOnly, showIndicators, verbose
    
    if addAuxPairs ==True:
        #trend mode
        shortTrendSignalTypes = bias
        shortModel=short_models[0]
        shortTrendPipelines=[
                       [shortModel],
                       #[fs_models[0],shortModel],
                       #[fs_models[1],shortModel],
                        ]
                        
        pv2e_SignalTypes =bias
        #pv2e_p2p_zz_std =long_zz_std
        pv2e_Model=auxPairs_models[0]
        pv2e_Pipelines=[
                        [pv2e_Model],
                        [fs_models[0],pv2e_Model],
                        #[fs_models[1],pv2e_p2pModel],
                        ]

        pv3s_SignalTypes = bias
        #pv3s_v2v_zz_std =long_zz_std
        pv3s_Model= auxPairs_models[0]               
        pv3s_Pipelines=[
                        [pv3s_Model],
                        [fs_models[0],pv3s_Model],
                        #[fs_models[1],pv3s_v2vModel],
                        ]
    else:
        #trend mode
        shortTrendSignalTypes = bias
        shortModel=short_models[0]
        shortTrendPipelines=[
                       [shortModel],
                       #[fs_models[0],shortModel],
                       #[fs_models[1],shortModel],
                        ]
                        
        pv2e_SignalTypes =bias
        #pv2e_p2p_zz_std =long_zz_std
        pv2e_Model=noAuxPairs_models[0]
        pv2e_Pipelines=[
                        [pv2e_Model],
                        #[fs_models[0],pv2e_Model],
                        #[fs_models[1],pv2e_p2pModel],
                        ]

        pv3s_SignalTypes = bias
        #pv3s_v2v_zz_std =long_zz_std
        pv3s_Model= noAuxPairs_models[0]               
        pv3s_Pipelines=[
                        [pv3s_Model],
                        #[fs_models[0],pv3s_Model],
                        #[fs_models[1],pv3s_v2vModel],
                        ]
                        
    #cycle mode pipelines same as trend mode
    shortCycleSignalTypes = shortTrendSignalTypes
    shortModel=short_models[0]
    shortCyclePipelines=shortTrendPipelines

    p2pSignalTypes =pv2e_SignalTypes
    #p2p_zz_std =long_zz_std
    p2pModel=pv2e_Model
    p2pPipelines=pv2e_Pipelines

    v2vSignalTypes = pv3s_SignalTypes
    #v2v_zz_std =long_zz_std
    v2vModel= pv3s_Model
    v2vPipelines=pv3s_Pipelines


    dataSet, auxPairsDict = loadCurrencyPairs(currencyPairs, dataPath,\
                                            barSizeSetting, maxlb, ticker,\
                                            signalPath, version, version_, maxReadLines,\
                                            perturbData=perturbData, perturbDataPct=perturbDataPct,\
                                            verbose=verbose, addAuxPairs=addAuxPairs)
    #account for data loss
    validationSetLength = dataSet.shape[0]-supportResistanceLB*2
    
    signalSets={
                    'wf_is_short':{},
                    'wf_is_pv2e_p2p':{},
                    'wf_is_pv3s_v2v':{},
                    }
    DpsRankByMetricB={
                    'best_wf_is_short':{},
                    'best_wf_is_pv2e_p2p':{},
                    'best_wf_is_pv3s_v2v':{},
                    'best_wf_is_all':{},
                    }
    DpsRankByMetricW={
                    'worst_wf_is_short':{},
                    'worst_wf_is_pv2e_p2p':{},
                    'worst_wf_is_pv3s_v2v':{},
                    'worst_wf_is_all':{},
                    }
    finalDF={}
    signalDF={}          
    stop=dataSet.shape[0]
    dataSet.index = dataSet.index.to_datetime()
    dataSet['gainAhead']=pd.Series(data=np.where(dataSet.Close.pct_change().shift(-1).values>0, 1,-1),index=dataSet.index)
    dataSet['gainAhead'].set_value(dataSet.index[-1],0)

    #for i in range(supportResistanceLB,dataSet.shape[0]):
    for start,i in enumerate(range(supportResistanceLB,stop-supportResistanceLB+1)):
        if start == supportResistanceLB or start==validationSetLength:
            if showCharts==False and showFinalChartOnly == True:
                showCharts=True
                showIndicators=True
                
        #maxlb is 2x support resistance lookback.  the first half is used to prime indicators
        data_primer = dataSet[start:maxlb+start]
        data_primer.index = data_primer.index.to_datetime()
        data_primer_ga = pd.Series(data=gainAhead(data_primer.Close),\
                                    index=data_primer.index, name='gainAhead')
        data_primer_ga_sig = pd.Series(data=np.where(gainAhead(data_primer.Close)>0,1,-1),\
                                    index=data_primer.index, name='gainAhead')
        dataSets={
                        'wf_is_short':pd.DataFrame(index=data_primer.index),
                        'wf_is_pv2e_p2p':pd.DataFrame(index=data_primer.index),
                        'wf_is_pv3s_v2v':pd.DataFrame(index=data_primer.index),
                        } 
                        
        #set the data to move one bar at a time. 
        data = dataSet[i:i+supportResistanceLB]
        nrows = data.shape[0]
        data.index = data.index.to_datetime()
        pv_sorted = []
        majorPeak=None
        majorValley=None
        minorPeak=None
        minorValley=None
        zz_std=outer_zz_std
        
        #modes = smoothHurst(data.Close, data.shape[0]-1,threshold=adfPvalue, showPlot=True)
        #pc = data.Close.pct_change().fillna(0)
        #zs=abs(pc[-1]-pc.mean())/pc.std()
        modes = mrClassifier(data.Close, data.shape[0]-1,threshold=adfPvalue, showPlot=debug,\
                                                   ticker=ticker)
        mode = modes[-1]
        if i ==supportResistanceLB:
            modePred = pd.Series(data=-1, index=data.index, name='volatilityPrediction')
            modePred.set_value(data.index[-1],mode)
        else:
            modePred.set_value(data.index[-1],mode)
            
        if mode ==0:
            #sort by small peaks
            reversePeaks = True
            reverseValleys = False
            #addAuxPairs = False
        else:
            #sort by large peaks
            reversePeaks = False
            reverseValleys = True
            #addAuxPairs = True
            
        #decrease stdev until there are three pivot points
        while majorPeak ==None or majorValley == None or minorPeak ==None or minorValley == None:
            zz = zg(data.Close,data.Close.pct_change().std()*zz_std,\
                        -data.Close.pct_change().std()*zz_std)
                        
            #data2 has integer index
            data2 = dataSet[i:i+supportResistanceLB].reset_index()
            peaks = [x for x  in np.where(zz.peak_valley_pivots()==1)[0]\
                            if supportResistanceLB-x>minDatapoints and x >0]
            peaksSorted=data2.Close.iloc[peaks].sort_values(ascending=reversePeaks).index
            if len(peaksSorted)>1:
                if mode==0:
                    minorPeak = peaksSorted[0]
                    plist=[peak for peak in peaksSorted if abs(minorPeak-peak) > minDatapoints]
                    if len(plist)>0:
                        #find closest peak to ensure single cycle
                        idx = (np.abs(np.array(plist)-minorPeak)).argmin()
                        majorPeak = plist[idx]
                else:
                    majorPeak = peaksSorted[0]
                    plist=[peak for peak in peaksSorted if abs(majorPeak-peak) > minDatapoints]
                    if len(plist)>0:
                        idx = (np.abs(np.array(plist)-majorPeak)).argmin()
                        minorPeak = plist[idx]
            #peaksSorted=data2.Close.iloc[peaks].sort_values(ascending=False).index
            #startPeak = peaksSorted[0]
            #minorPeak = [peak for peak in peaksSorted if abs(startPeak-peak) > minDatapoints][0]

            valleys =  [x for x  in np.where(zz.peak_valley_pivots()==-1)[0]\
                                if supportResistanceLB-x>minDatapoints and x >0]
            valleysSorted = data2.Close.iloc[valleys].sort_values(ascending=reverseValleys).index
            if len(valleysSorted)>1:
                if mode==0:
                    minorValley = valleysSorted[0]
                    vlist = [valley for valley in valleysSorted if abs(minorValley-valley) > minDatapoints]
                    if len(vlist)>0:
                        #find closest valley to ensure single cycle
                        idx = (np.abs(np.array(vlist)-minorValley)).argmin()
                        majorValley = vlist[idx]
                else:
                    majorValley = valleysSorted[0]
                    vlist = [valley for valley in valleysSorted if abs(majorValley-valley) > minDatapoints]
                    if len(vlist)>0:
                        idx = (np.abs(np.array(vlist)-majorValley)).argmin()
                        minorValley = vlist[idx]    
            #valleysSorted = data2.Close.iloc[valleys].sort_values(ascending=True).index
            #startValley = valleysSorted[0]
            #minorValley = [valley for valley in valleysSorted if abs(startValley-valley) > minDatapoints][0]

            #shortStart=[x for x  in np.where(zz.peak_valley_pivots()!=0)[0]\
            #                   if supportResistanceLB-x>minDatapoints][-1]
            
            #pv2e_p2p_period=sorted([supportResistanceLB-startPeak, supportResistanceLB-minorPeak])
            #pv3s_v2v_period=sorted([supportResistanceLB-startValley, supportResistanceLB-minorValley])
            pv_sorted = np.asarray(sorted(peaks+valleys))
            zz_std=zz_std*.9
            
        halfCycles = np.diff(pv_sorted).tolist()
        cycleList = [[halfCycles[j], (x,data2.Close[x])] for j,x in \
                                                    enumerate(sorted(valleys+peaks)[1:])]
        #zz.plot_pivots(cycleList=cycleList)
        #shBars = average(np.array(halfCycles)*2)

            
        train_index = []
        if mode==0:
        #0 cycle mode
            #calc peaks and valleys
            #peaksSorted=data2.Close.iloc[peaks].sort_values(ascending=False).index
            #majorPeak = peaksSorted[0]
            #minorPeak = [peak for peak in peaksSorted if abs(majorPeak-peak) > minDatapoints][0]
            #valleysSorted = data2.Close.iloc[valleys].sort_values(ascending=True).index
            #majorValley = valleysSorted[0]
            #minorValley = [valley for valley in valleysSorted if abs(majorValley-valley) > minDatapoints][0]    
            shortStart=[x for x  in np.where(zz.peak_valley_pivots()!=0)[0]\
                                if supportResistanceLB-x>minDatapoints][-1]
            short_period =supportResistanceLB- shortStart
            p2p_period=sorted([supportResistanceLB-majorPeak, supportResistanceLB-minorPeak])
            v2v_period=sorted([supportResistanceLB-majorValley, supportResistanceLB-minorValley])
            
            ytrain1=zz.pivots_to_modes()[-len(data2.index[-short_period:-1]):]
            ytrain2=zz.pivots_to_modes()[data2.index[-p2p_period[1]:-p2p_period[0]+1]+1]
            ytrain3=zz.pivots_to_modes()[data2.index[-v2v_period[1]:-v2v_period[0]+1]+1]
            train_index = [
                            #-1 to exclude last index for y_train
                            ('wf_is_short',ytrain1,\
                                    data2.index[-short_period:-1],shortCyclePipelines,\
                                    shortCycleSignalTypes),
                            #+1 to include bottom of valley/top of the peak
                            ('wf_is_pv2e_p2p',ytrain2,\
                                    data2.index[-p2p_period[1]:-p2p_period[0]+1],\
                                    p2pPipelines,p2pSignalTypes),
                            ('wf_is_pv3s_v2v', ytrain3,\
                                    data2.index[-v2v_period[1]:-v2v_period[0]+1],\
                                    v2vPipelines,v2vSignalTypes)
                            ]
                            
        else:
        #1 trend mode
            shortStart = pv_sorted[-1]
            short_period =supportResistanceLB- shortStart
            is_period='wf_is_short'
            index = data2.index[-short_period:-1]
            #spike mode
            #y_train_zz = np.where(zz.pivots_to_modes()[-len(index):]<0,1,-1)
            #non-spike mode
            y_train_zz =zz.pivots_to_modes()[-len(index):]
            signal_types=shortTrendSignalTypes
            pipelines=shortTrendPipelines
            train_index.append((is_period,y_train_zz,index,pipelines,signal_types))
            
            pv2=3
            if pv_sorted[-pv2] in peaks:
                startPeak=pv_sorted[-pv2]
                #-1 to exclude last index for y_train
                pv2e_period=range(pv_sorted[-pv2],supportResistanceLB-1)
                is_period='wf_is_pv2e_p2p'
                index = data2.index[pv2e_period]
                y_train_zz = zz.pivots_to_modes()[-len(index):]
                signal_types=pv2e_SignalTypes
                pipelines=pv2e_Pipelines
                train_index.append((is_period,y_train_zz,index,pipelines,signal_types))
            else:
                startValley=pv_sorted[-pv2]
                #-1 to exclude last index for y_train
                pv2e_period=range(pv_sorted[-pv2],supportResistanceLB-1)
                is_period='wf_is_pv2e_p2p'
                index = data2.index[pv2e_period]
                y_train_zz = zz.pivots_to_modes()[-len(index):]
                signal_types=pv2e_SignalTypes
                pipelines=pv2e_Pipelines
                train_index.append((is_period,y_train_zz,index,pipelines,signal_types))
            
            pv3=4
            if pv_sorted[-pv3] in peaks:
                startPeak=pv_sorted[-pv3]
                #+1 because range starts from 0 index
                #pv3s_v2v_period=range(pv_sorted[-3]+1,shortStart+1)
                #pv3s_v2v_period=range(pv_sorted[-3],shortStart)
                is_period='wf_is_pv3s_v2v'
                index = data2.index[-supportResistanceLB-pv_sorted[-3]:-short_period+1]
                y_train_zz = zz.pivots_to_modes()[index+1]
                signal_types=pv3s_SignalTypes
                pipelines=pv3s_Pipelines
                train_index.append((is_period,y_train_zz,index,pipelines,signal_types))
            else:
                startValley=pv_sorted[-pv3]
                #+1 because range starts from 0 index
                #pv3s_v2v_period=range(pv_sorted[-3]+1,shortStart+1)
                #pv3s_v2v_period=range(pv_sorted[-3],shortStart)
                is_period='wf_is_pv3s_v2v'
                index = data2.index[-supportResistanceLB-pv_sorted[-3]:-short_period+1]
                y_train_zz = zz.pivots_to_modes()[index+1]
                signal_types=pv3s_SignalTypes
                pipelines=pv3s_Pipelines
                train_index.append((is_period,y_train_zz,index,pipelines,signal_types))
            
        


        windowLengths={}
        for is_period, y_train_zz, index, pipelines, signal_types in train_index:
            lb = len(index)

            inner_pv_sorted=[]
            zz_std = inner_zz_std
            while len(inner_pv_sorted)<3 and zz_std>0.5:
                #print len(inner_pv_sorted),
                zz_inner=zg(data.iloc[index].Close,data.iloc[index].Close.pct_change().std()*zz_std,\
                        -data.iloc[index].Close.pct_change().std()*zz_std)
                inner_peaks = [x for x  in np.where(zz_inner.peak_valley_pivots()==1)[0]]
                inner_valleys = [x for x  in np.where(zz_inner.peak_valley_pivots()==-1)[0]]
                inner_pv_sorted = np.asarray(sorted(inner_peaks+inner_valleys))
                inner_halfCycles = np.diff(inner_pv_sorted).tolist()
                zz_std=zz_std*0.7
                #print len(inner_pv_sorted)
                #if len(index)-minDatapoints<3:
                #    break
            
            #x2 for full cycle length
            is_cycle=average(np.array(inner_halfCycles)*2)
            #correlation < 2 creates inf
            if is_cycle < 2:
                is_cycle =2
                
            windowLengths[is_period] = is_cycle

            #short indicators
            dataSets[is_period][ticker+'_Pri_RSI_c'+str(1.5)] =\
                                                RSI(data_primer.Close,1.5)
            #dataSets[is_period]['Pri_KE_c'+str(is_cycle)+'_r'+str(lb)] = roofingFilter(kaufman_efficiency(data_primer.Close,\
            #                                                    is_cycle),
            #                                                    supportResistanceLB,lb)
            #dataSets[is_period]['Pri_DPO_c'+str(is_cycle)+'_r'+str(lb)] = zScore(DPO(data_primer.Close,\
            #                                                        is_cycle),lb)
            #dataSets[is_period]['ZS_priceChange_c'+str(is_cycle)+'_r'+str(lb)] = zScore(priceChange(\
            #                            data_primer.Close),lb)
            #dataSets[is_period]['mean_pc_c'+str(is_cycle)+'_r'+str(lb)] = priceChange(data_primer.Close-\
            #                                        pd.rolling_mean(priceChange(\
            #                                    data_primer.Close),lb))
            
            #long indicators
            #dataSets[is_period]['Pri_ROC_c'+str(is_cycle)+'_r'+str(lb)] = ROC(data_primer.Close,supportResistanceLB)
            dataSets[is_period][ticker+'_Pri_rStoch_r'+str(lb)] = \
                                                                        roofingFilter(data_primer.Close,\
                                                                                    supportResistanceLB,lb)
            
            #volatility
            if is_cycle==supportResistanceLB:
                #hot fix for insufficient lookback
                is_cycle_atr=supportResistanceLB/2
            else:
                is_cycle_atr=is_cycle
            dataSets[is_period][ticker+'_Pri_ATR_c'+str(is_cycle)+'_r'+str(lb)] = \
                                                            roofingFilter(ATR(data_primer.High,
                                                            data_primer.Low,
                                                            data_primer.Close,is_cycle),
                                                            supportResistanceLB,lb)


            #correlations
            dataSets[is_period][ticker+'_Pri_autoCor1_c'+str(is_cycle)+'_r'+str(lb)] =\
                                                    roofingFilter(autocorrel(data_primer.Close*100,\
                                                                                is_cycle,period=1),
                                                                        supportResistanceLB,lb)

            
            indicator_df = dataSets[is_period].iloc[-supportResistanceLB:].iloc[index]

            if addAuxPairs:
                for pair in auxPairsDict:
                    auxPairdataSet = pd.DataFrame(index=data_primer.index)
                    closes=auxPairsDict[pair]['closes'].iloc[:,1][:maxlb]
                    '''
                    highs=auxPairsDict[pair]['closes'].iloc[:,1][:maxlb]
                    lows=auxPairsDict[pair]['closes'].iloc[:,1][:maxlb]
                    auxPairdataSet[pair+'_Pri_RSI_c'+str(is_cycle)+'_r'+str(lb)] =\
                                                    RSI(closes,is_cycle)
                    auxPairdataSet[pair+'_Pri_Corr1_c'+str(is_cycle)+'_r'+str(lb)] = \
                                                    roofingFilter(pd.Series(pd.rolling_corr(\
                                                    data_primer.Close.shift(1).values*100,\
                                                    closes.values*100, window=is_cycle))\
                                                    .replace([np.inf, -np.inf, np.nan],[1.0,-1.0,0]).values,\
                                                    supportResistanceLB,lb)
                    auxPairdataSet[pair+'_Pri_rStoch_c'+str(is_cycle)+'_r'+str(lb)] =\
                                            roofingFilter(closes,supportResistanceLB,lb)
                    auxPairdataSet[pair+'_Pri_ATR_c'+str(is_cycle)+'_r'+str(lb)] =\
                                                                    roofingFilter(ATR(highs,lows,\
                                                            closes,is_cycle), supportResistanceLB,lb)
                    '''
                    auxPairdataSet[pair+'Pri_ROC_c'+str(is_cycle)] =\
                                        ROC(closes,is_cycle)
                    dataSets[is_period] = pd.concat([dataSets[is_period], auxPairdataSet], axis=1)
                    #auxPairdataSet.iloc[-supportResistanceLB:].iloc[index].plot()
                
            DominantCycle, acp = acPeriodogram(data_primer.Close,bars=is_cycle)
            dataSets[is_period] = pd.concat([dataSets[is_period], acp], axis=1)
            lastIndex = dataSets[is_period].index[-1]
            if verbose:
                if mode==0:
                    print '\n'+ticker+' '+is_period+' Low Volatility: Cycle Mode '+str(lastIndex)
                else:
                    print '\n'+ticker+' '+is_period+' High Volatility Trend Mode '+str(lastIndex)
                print 'index',i,'cols',dataSets[is_period].shape[1],'nrows (lb)' ,lb,'ZZ Cycle',is_cycle,'AC Cycle',DominantCycle
                
            for st in signal_types:
                # ['gainAhead','zigZag','buyHold','sellHold']
                x_train = dataSets[is_period].iloc[-supportResistanceLB:].iloc[index].values
                #print np.where(x_train<0)
                x_test = dataSets[is_period].values[-1]
                
                #print lastIndex
                if st == 'buyHold':
                    dictName = st
                    if i == supportResistanceLB:
                        signalSets[is_period][dictName]=initSST(data.Close, version_, st, initialEquity)
                    
                    signalSets[is_period][dictName].set_value(lastIndex, 'signals', 1)
                    signalSets[is_period][dictName].set_value(lastIndex, 'signalType', dictName)
                    signalSets[is_period][dictName].set_value(lastIndex, version_+'_system', st)
                    signalSets[is_period][dictName].set_value(dataSets[is_period].index[-2],'gainAhead',data.Close[-2:].pct_change()[-1])
                    signalSets[is_period][dictName].set_value(lastIndex,'gainAhead',0)
                    signalSets[is_period][dictName].set_value(lastIndex,'nodpsSafef',nodpsSafef)
                    netPNL, commission, lastEquity = calcEquityLast(-1, signalSets[is_period][dictName])
                    signalSets[is_period][dictName].set_value(lastIndex,'netPNL',netPNL)
                    signalSets[is_period][dictName].set_value(lastIndex,'nodpsComm',commission)
                    signalSets[is_period][dictName].set_value(lastIndex,'netEquity',lastEquity)

                elif st == 'sellHold':
                    dictName = st
                    if i == supportResistanceLB:
                        signalSets[is_period][dictName]=initSST(data.Close, version_, st, initialEquity)
                    
                    signalSets[is_period][dictName].set_value(lastIndex, 'signals', -1)
                    signalSets[is_period][dictName].set_value(lastIndex, 'signalType', dictName)
                    signalSets[is_period][dictName].set_value(lastIndex, version_+'_system', st)
                    signalSets[is_period][dictName].set_value(dataSets[is_period].index[-2],'gainAhead',data.Close[-2:].pct_change()[-1])
                    signalSets[is_period][dictName].set_value(lastIndex,'gainAhead',0)
                    signalSets[is_period][dictName].set_value(lastIndex,'nodpsSafef',nodpsSafef)
                    netPNL, commission, lastEquity = calcEquityLast(-1, signalSets[is_period][dictName])
                    signalSets[is_period][dictName].set_value(lastIndex,'netPNL',netPNL)
                    signalSets[is_period][dictName].set_value(lastIndex,'nodpsComm',commission)
                    signalSets[is_period][dictName].set_value(lastIndex,'netEquity',lastEquity)
                        
                elif st =='gainAhead':
                    reverse='normal'
                    y_train_ga = data_primer_ga_sig.iloc[-supportResistanceLB:].iloc[index].values
                    #reverse if choppy bias, short is and cycle (cycle peak bias)
                    if is_period=='wf_is_short' and mode==0:
                        reverse='reversed'
                        y_train_ga = np.where(y_train_ga<0,1,-1)

                        
                    if verbose:
                        print st, reverse,y_train_ga
                    for pipeline in pipelines:
                        if len(pipeline) >1:
                            cols = nfeatures
                        else:
                            cols = x_train.shape[1]
                        if len(pipeline) ==2:
                            dictName = st+'_'+pipeline[0][0]+'_'+pipeline[1][0]
                        else:
                            dictName = st+'_'+pipeline[0][0]
                        #init
                        if i == supportResistanceLB:
                            signalSets[is_period][dictName]=initSST(data.Close, version_, st, initialEquity)
                            
                        #train/predict    
                        Pipeline(pipeline).fit(x_train, y_train_ga)
                        signal = Pipeline(pipeline).predict([x_test])
                        col_name = st+'_'+pipeline[0][0]+'_f'+str(cols)+'_is'+str(lb)
                        #print signal,dataSet.ix[lastIndex].gainAhead,col_name
                        
                        #append to df
                        #signalSets[is_period][dictName].set_value(lastIndex,col_name , signal)
                        signalSets[is_period][dictName].set_value(lastIndex, 'signals', signal)
                        signalSets[is_period][dictName].set_value(lastIndex, 'signalType', dictName)
                        signalSets[is_period][dictName].set_value(lastIndex, version_+'_system', col_name)
                        signalSets[is_period][dictName].set_value(dataSets[is_period].index[-2],'gainAhead',data.Close[-2:].pct_change()[-1])
                        signalSets[is_period][dictName].set_value(lastIndex,'gainAhead',0)
                        signalSets[is_period][dictName].set_value(lastIndex,'nodpsSafef',nodpsSafef)
                        netPNL, commission, lastEquity = calcEquityLast(-1, signalSets[is_period][dictName])
                        signalSets[is_period][dictName].set_value(lastIndex,'netPNL',netPNL)
                        signalSets[is_period][dictName].set_value(lastIndex,'nodpsComm',commission)
                        signalSets[is_period][dictName].set_value(lastIndex,'netEquity',lastEquity)
                        if verbose:
                            print signal,dataSet.ix[lastIndex].gainAhead,col_name
                            print 'prior signal', signalSets[is_period][dictName].signals[-2], signalSets[is_period][dictName].gainAhead[-2], netPNL, commission, lastEquity

                elif st == 'zigZag':
                    reverse='normal'
                    #reverse if choppy bias, short is and cycle (cycle peak bias)
                    if is_period=='wf_is_short' and mode==0:
                        reverse='reversed'
                        y_train_zz = np.where(y_train_zz<0,1,-1)
                    #reverse if choppy bias, pv3s is and trend (cycle peak bias)
                    #if bias[0]=='gainAhead' and is_period=='wf_is_pv3s_v2v' and mode==1:
                    #    reverse='reversed'
                    #    y_train_ga = np.where(y_train_ga<0,1,-1)
                        
                    #ytrain at beginnign of for loop
                    if verbose:
                        print st, reverse, y_train_zz
                    for pipeline in pipelines:
                        if len(pipeline) >1:
                            cols = nfeatures
                        else:
                            cols = x_train.shape[1]
                            
                        if len(pipeline) ==2:
                            dictName = st+'_'+pipeline[0][0]+'_'+pipeline[1][0]
                        else:
                            dictName = st+'_'+pipeline[0][0]
                        #init
                        if i == supportResistanceLB:
                            signalSets[is_period][dictName]=initSST(data.Close, version_, st, initialEquity)
                            
                        #train/predict    
                        Pipeline(pipeline).fit(x_train, y_train_zz)
                        signal = Pipeline(pipeline).predict([x_test])
                        col_name= st+'_'+pipeline[0][0]+'_f'+ str(cols)+'_is'+str(lb)
                        
                        
                        #append to df
                        #signalSets[is_period][dictName].set_value(lastIndex, col_name, signal)
                        signalSets[is_period][dictName].set_value(lastIndex, 'signals', signal)
                        signalSets[is_period][dictName].set_value(lastIndex, 'signalType', dictName)
                        signalSets[is_period][dictName].set_value(lastIndex, version_+'_system', col_name)
                        signalSets[is_period][dictName].set_value(dataSets[is_period].index[-2],'gainAhead',data.Close[-2:].pct_change()[-1])
                        signalSets[is_period][dictName].set_value(lastIndex,'gainAhead',0)
                        signalSets[is_period][dictName].set_value(lastIndex,'nodpsSafef',nodpsSafef)
                        netPNL, commission, lastEquity = calcEquityLast(-1, signalSets[is_period][dictName])
                        signalSets[is_period][dictName].set_value(lastIndex,'netPNL',netPNL)
                        signalSets[is_period][dictName].set_value(lastIndex,'nodpsComm',commission)
                        signalSets[is_period][dictName].set_value(lastIndex,'netEquity',lastEquity)
                        if verbose:
                            print signal,dataSet.ix[lastIndex].gainAhead,col_name
                            print 'prior signal', signalSets[is_period][dictName].signals[-2], signalSets[is_period][dictName].gainAhead[-2], netPNL, commission, lastEquity

            
            #show zz chart  
            if showIndicators:
                cycleList2 = [[inner_halfCycles[j], (x,data.iloc[-supportResistanceLB:].iloc[index].Close[x])] for j,x in \
                                            enumerate(sorted(inner_valleys+inner_peaks)[1:])]
                zz_inner.plot_pivots(l=8,w=8, cycleList=cycleList2,mode=modePred.ix[data.iloc[index].index],\
                                               indicators=indicator_df, chartTitle=ticker+' '+is_period,\
                                               savePath=chartSavePath+'_TECH_'+is_period, debug=debug)
                                               
        dpsDF_all = pd.DataFrame()
        #set to lb for stability
        #windowLength = supportResistanceLB
        for is_period in signalSets:
            dpsDF = pd.DataFrame()
            
            for col in signalSets[is_period]:    
                #calcDPS
                windowLength = windowLengths[is_period]
                if verbose:
                    print 'Using window length',windowLength, 'for', is_period
                if PRT is not None and windowLength is not None:
                    dps = calcDPS(col, signalSets[is_period][col], PRT, windowLength, verbose=False,\
                                            asset=asset)
                    dps['is_period']=is_period
                    if i == supportResistanceLB:
                        signalSets[is_period][col]=signalSets[is_period][col][:-1].append(dps).fillna(0)
                        signalSets[is_period][col].set_value(signalSets[is_period][col][:-1].index,'dpsNetEquity',initialEquity)
                    else:
                        signalSets[is_period][col]=signalSets[is_period][col][:-1].append(dps)
                        
                #append last dps result for ranking
                dpsDF = dpsDF.append(dps)
                dpsDF_all = dpsDF_all.append(dps)
                
            #save top ranks by is period
            if i == supportResistanceLB:
                DpsRankByMetricB['best_'+is_period] = signalSets[is_period]\
                                                                [dpsDF.sort_values(by=metric, ascending=False).iloc[0]\
                                                                ['signalType']].copy(deep=True)
               
                DpsRankByMetricW['worst_'+is_period]= signalSets[is_period]\
                                                                [dpsDF.sort_values(by=metric, ascending=False).iloc[-1]\
                                                                ['signalType']].copy(deep=True)           
            else:
                #print dpsDF
                #best CAR25
                DpsRankByMetricB['best_'+is_period]=DpsRankByMetricB['best_'+is_period].append(dpsDF.sort_values(by=metric, ascending=False).iloc[0])
                #worstCAR25
                DpsRankByMetricW['worst_'+is_period]=DpsRankByMetricW['worst_'+is_period].append(dpsDF.sort_values(by=metric, ascending=False).iloc[-1])
                #best model
            
        #total ranking    
        if i == supportResistanceLB:
            is_period = dpsDF_all.sort_values(by=metric, ascending=False).iloc[0]['is_period']
            signalType = dpsDF_all.sort_values(by=metric, ascending=False).iloc[0]['signalType']
            DpsRankByMetricB['best_wf_is_all'] = signalSets[is_period][signalType].copy(deep=True)
            
            is_period = dpsDF_all.sort_values(by=metric, ascending=False).iloc[-1]['is_period']
            signalType = dpsDF_all.sort_values(by=metric, ascending=False).iloc[-1]['signalType']
            DpsRankByMetricW['worst_wf_is_all']= signalSets[is_period][signalType].copy(deep=True)
            
        else:
            #print dpsDF
            #best CAR25
            DpsRankByMetricB['best_wf_is_all']=DpsRankByMetricB['best_wf_is_all'].append(dpsDF_all.sort_values(by=metric, ascending=False).iloc[0])
            #worstCAR25
            DpsRankByMetricW['worst_wf_is_all']=DpsRankByMetricW['worst_wf_is_all'].append(dpsDF_all.sort_values(by=metric, ascending=False).iloc[-1])
            #best model    
            

        
        #print  len(signalSets[is_period][col]), 'mod0', len(signalSets[is_period][col])%supportResistanceLB, 'end', stop-supportResistanceLB-1
        #if len(signalSets[is_period][col])%supportResistanceLB ==0 or i==stop-supportResistanceLB-1:
        dpsDF_all2 = pd.DataFrame()
        dpsDF_BofB = pd.DataFrame()
        dpsDF_BofW = pd.DataFrame()
        
        #windowLength = int(np.array([windowLengths[x] for x in windowLengths]).mean())
        windowLength = supportResistanceLB
        if verbose:
            print '\nwindowLength  for final EC calc', windowLength
        for rank in DpsRankByMetricB:
            DpsRankByMetricB[rank].index.name = 'dates'
            updateDps = calcDPS(rank, DpsRankByMetricB[rank], PRT, windowLength, verbose=False,\
                                            asset=asset)
            dpsDF_all2 = dpsDF_all2.append(updateDps)
            dpsDF_BofB = dpsDF_BofB.append(updateDps)
            DpsRankByMetricB[rank].set_value(updateDps.index, updateDps.columns, updateDps.values)
            index2 = DpsRankByMetricB[rank].index.intersection(data_primer_ga.index)
            DpsRankByMetricB[rank].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            DpsRankByMetricB[rank]=reCalcEquity(DpsRankByMetricB[rank], metric)

        for rank in DpsRankByMetricW:
            DpsRankByMetricW[rank].index.name = 'dates'
            updateDps = calcDPS(rank, DpsRankByMetricW[rank], PRT, windowLength, verbose=False,\
                                            asset=asset)
            dpsDF_all2 = dpsDF_all2.append(updateDps)
            dpsDF_BofW = dpsDF_BofW.append(updateDps)
            DpsRankByMetricW[rank].set_value(updateDps.index, updateDps.columns, updateDps.values)
            index2 = DpsRankByMetricW[rank].index.intersection(data_primer_ga.index)
            DpsRankByMetricW[rank].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            DpsRankByMetricW[rank]=reCalcEquity(DpsRankByMetricW[rank], metric)
        
        if i == supportResistanceLB:
            finalDF['finalBest']=pd.DataFrame(index=data.index)
            finalDF['finalBest'].set_value(finalDF['finalBest'].index,'dpsNetEquity',initialEquity)
            finalDF['finalBest'].set_value(finalDF['finalBest'].index,'netEquity',initialEquity)
            finalDF['finalBest']=finalDF['finalBest'][:-1].append(dpsDF_all2.sort_values(by=metric, ascending=False).iloc[0]).fillna(0)

            finalDF['finalWorst']=pd.DataFrame(index=data.index)
            finalDF['finalWorst'].set_value(finalDF['finalWorst'].index,'dpsNetEquity',initialEquity)
            finalDF['finalWorst'].set_value(finalDF['finalWorst'].index,'netEquity',initialEquity)
            finalDF['finalWorst']=finalDF['finalWorst'][:-1].append(dpsDF_all2.sort_values(by=metric, ascending=True).iloc[0]).fillna(0)
            
            finalDF['finalBestOfBest']=pd.DataFrame(index=data.index)
            finalDF['finalBestOfBest'].set_value(finalDF['finalBestOfBest'].index,'dpsNetEquity',initialEquity)
            finalDF['finalBestOfBest'].set_value(finalDF['finalBestOfBest'].index,'netEquity',initialEquity)
            finalDF['finalBestOfBest']=finalDF['finalBestOfBest'][:-1].append(dpsDF_BofB.sort_values(by=metric, ascending=False).iloc[0]).fillna(0)

            finalDF['finalBestOfWorst']=pd.DataFrame(index=data.index)
            finalDF['finalBestOfWorst'].set_value(finalDF['finalBestOfWorst'].index,'dpsNetEquity',initialEquity)
            finalDF['finalBestOfWorst'].set_value(finalDF['finalBestOfWorst'].index,'netEquity',initialEquity)
            finalDF['finalBestOfWorst']=finalDF['finalBestOfWorst'][:-1].append(dpsDF_BofW.sort_values(by=metric, ascending=False).iloc[0]).fillna(0)

        else:
            finalDF['finalBest']=finalDF['finalBest'].append(dpsDF_all2.sort_values(by=metric, ascending=False).iloc[0])
            index2 = finalDF['finalBest'].index.intersection(data_primer_ga.index)
            finalDF['finalBest'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            finalDF['finalBest']=reCalcEquity(finalDF['finalBest'], metric)
            
            finalDF['finalWorst']=finalDF['finalWorst'].append(dpsDF_all2.sort_values(by=metric, ascending=True).iloc[0])
            index2 = finalDF['finalWorst'].index.intersection(data_primer_ga.index)
            finalDF['finalWorst'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            finalDF['finalWorst']=reCalcEquity(finalDF['finalWorst'], metric)

            finalDF['finalBestOfBest']=finalDF['finalBestOfBest'].append(dpsDF_BofB.sort_values(by=metric, ascending=False).iloc[0])
            index2 = finalDF['finalBestOfBest'].index.intersection(data_primer_ga.index)
            finalDF['finalBestOfBest'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            finalDF['finalBestOfBest']=reCalcEquity(finalDF['finalBestOfBest'], metric)
            
            finalDF['finalBestOfWorst']=finalDF['finalBestOfWorst'].append(dpsDF_BofW.sort_values(by=metric, ascending=False).iloc[0])
            index2 = finalDF['finalBestOfWorst'].index.intersection(data_primer_ga.index)
            finalDF['finalBestOfWorst'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
            finalDF['finalBestOfWorst']=reCalcEquity(finalDF['finalBestOfWorst'], metric)
            
        dpsDF_final = pd.DataFrame()
        for rank in finalDF:
            finalDF[rank].index.name = 'dates'
            updateDps = calcDPS(rank, finalDF[rank], PRT, windowLength, verbose=False,\
                                            asset=asset)
            dpsDF_final = dpsDF_final.append(updateDps)
            finalDF[rank].set_value(updateDps.index, updateDps.columns, updateDps.values)
        '''
        #mean reversion
        if mode==0:   
            #cycle
            mr=False
            metric2='dpsROC'
            curve='dpsROC'
            #metric2=metric
            #curve='WorstCAR25 of reDPS B/W'
            if verbose:
                print 'Cycle mode - choosing signal from', curve
        else:
            #trend - select from the worst
            mr=True
            metric2='dpsROC'
            curve='dpsROC'        
            #metric2=metric
            #curve='WorstCAR25 of reDPS B/W'
            if verbose:
                print 'Trend mode - choosing signal from', curve

            '''
        #if mode==0:   
        #   mr=True
        #    #metric2='netEquity'
        #    curve='lowest '+metric2
        #else:
        #dpsDF_all2 = dpsDF_all2.append(dpsDF_all)
        #dpsDF_all2 = dpsDF_all2.append(dpsDF_final)
        
        if mode ==0:
            mr=True
            curve='lowest '+metric2
            if i == supportResistanceLB:
                #dpsDF_final['nodpsROC']=dpsDF_final.netPNL/dpsDF_final.netEquity
                #dpsDF_final['dpsROC']=dpsDF_final.dpsNetPNL/dpsDF_final.dpsNetEquity
                signalDF['tripleFiltered']=pd.DataFrame(index=data.index)
                signalDF['tripleFiltered'].set_value(signalDF['tripleFiltered'].index,'dpsNetEquity',initialEquity)
                signalDF['tripleFiltered'].set_value(signalDF['tripleFiltered'].index,'netEquity',initialEquity)
                signalDF['tripleFiltered']=signalDF['tripleFiltered'][:-1].append(dpsDF_all2.sort_values(by=metric2, ascending=mr).iloc[0]).fillna(0)
            else:
                #dpsDF_final['nodpsROC']=dpsDF_final.netPNL/dpsDF_final.netEquity
                #dpsDF_final['dpsROC']=dpsDF_final.dpsNetPNL/dpsDF_final.dpsNetEquity
                signalDF['tripleFiltered']=signalDF['tripleFiltered'].append(dpsDF_all2.sort_values(by=metric2, ascending=mr).iloc[0])
                index2 = signalDF['tripleFiltered'].index.intersection(data_primer_ga.index)
                signalDF['tripleFiltered'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
                signalDF['tripleFiltered']=reCalcEquity(signalDF['tripleFiltered'], metric2)
        else:
            mr=False
            curve='highest '+metric
            if i == supportResistanceLB:
                #dpsDF_final['nodpsROC']=dpsDF_final.netPNL/dpsDF_final.netEquity
                #dpsDF_final['dpsROC']=dpsDF_final.dpsNetPNL/dpsDF_final.dpsNetEquity
                signalDF['tripleFiltered']=pd.DataFrame(index=data.index)
                signalDF['tripleFiltered'].set_value(signalDF['tripleFiltered'].index,'dpsNetEquity',initialEquity)
                signalDF['tripleFiltered'].set_value(signalDF['tripleFiltered'].index,'netEquity',initialEquity)
                signalDF['tripleFiltered']=signalDF['tripleFiltered'][:-1].append(dpsDF_all.sort_values(by=metric, ascending=mr).iloc[0]).fillna(0)
            else:
                #dpsDF_final['nodpsROC']=dpsDF_final.netPNL/dpsDF_final.netEquity
                #dpsDF_final['dpsROC']=dpsDF_final.dpsNetPNL/dpsDF_final.dpsNetEquity
                signalDF['tripleFiltered']=signalDF['tripleFiltered'].append(dpsDF_all.sort_values(by=metric, ascending=mr).iloc[0])
                index2 = signalDF['tripleFiltered'].index.intersection(data_primer_ga.index)
                signalDF['tripleFiltered'].set_value(index2,'gainAhead',data_primer_ga.ix[index2].values) 
                signalDF['tripleFiltered']=reCalcEquity(signalDF['tripleFiltered'], metric)
        #print dpsDF_final.sort_values(by=metric2, ascending=mr).iloc[0]
        #print signalDF['tripleFiltered'].iloc[-1]
                
        if showCharts:                
            zz.plot_pivots(l=8,w=8,\
                                #startValley=(startValley, data2.Close[startValley]),\
                                #startPeak=(startPeak, data2.Close[startPeak]),\
                                #minorValley=(minorValley, data2.Close[minorValley]),\
                                #minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                #shortStart=(shortStart, data2.Close[shortStart]),\
                                cycleList=cycleList,mode=modePred[start:],\
                                signals=finalDF,chartTitle=ticker+' reDPS of B/W by '+metric+' final curves',\
                                savePath=chartSavePath+'_FINAL', debug=debug
                                )
                                
            if mode==0:
            #cycle mode
                zz.plot_pivots(l=8,w=8,\
                                    majorValley=(majorValley, data2.Close[majorValley]),\
                                    majorPeak=(majorPeak, data2.Close[majorPeak]),\
                                    minorValley=(minorValley, data2.Close[minorValley]),\
                                    minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                    shortStart=(shortStart, data2.Close[shortStart]),\
                                    cycleList=cycleList,mode=modePred[start:],\
                                    signals=DpsRankByMetricB,chartTitle=ticker+' Best Rank '+metric,\
                                    savePath=chartSavePath+'_BRANK', debug=debug
                                    )
                                    
                zz.plot_pivots(l=8,w=8,\
                                    majorValley=(majorValley, data2.Close[majorValley]),\
                                    majorPeak=(majorPeak, data2.Close[majorPeak]),\
                                    minorValley=(minorValley, data2.Close[minorValley]),\
                                    minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                    shortStart=(shortStart, data2.Close[shortStart]),\
                                    cycleList=cycleList,mode=modePred[start:],\
                                    signals=DpsRankByMetricW,chartTitle=ticker+' Worst Rank '+metric,\
                                    savePath=chartSavePath+'_WRANK', debug=debug
                                    )    
            else:
            #trend mode
                zz.plot_pivots(l=8,w=8,\
                                    startValley=(startValley, data2.Close[startValley]),\
                                    startPeak=(startPeak, data2.Close[startPeak]),\
                                    #minorValley=(minorValley, data2.Close[minorValley]),\
                                    #minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                    shortStart=(shortStart, data2.Close[shortStart]),\
                                    cycleList=cycleList,mode=modePred[start:],\
                                    signals=DpsRankByMetricB,chartTitle=ticker+' Best Rank '+metric,\
                                    savePath=chartSavePath+'_BRANK', debug=debug
                                    )
                                    
                zz.plot_pivots(l=8,w=8,\
                                    startValley=(startValley, data2.Close[startValley]),\
                                    startPeak=(startPeak, data2.Close[startPeak]),\
                                    #minorValley=(minorValley, data2.Close[minorValley]),\
                                    #minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                    shortStart=(shortStart, data2.Close[shortStart]),\
                                    cycleList=cycleList,mode=modePred[start:],\
                                    signals=DpsRankByMetricW,chartTitle=ticker+' Worst Rank '+metric,\
                                    savePath=chartSavePath+'_WRANK', debug=debug
                                    )       
                                
            for is_period in signalSets:
                if verbose:
                    print 'nrows', nrows, 'idx', i
                            #,'minor v',minorValley,'minor p',minorPeak,\
                #print 'valleys', valleys
                #print 'peaks', peaks
                #print 'half cycles', halfCycles
                #print is_period, 'wf_is_periods', wf_is_periods, 'wfSteps', wfSteps
                #print 'is_cycle_long', is_cycle_long, 'is_cycle_mid', is_cycle_mid, 'is_cycle_short', is_cycle_short
                if mode==0:
                    zz.plot_pivots(l=8,w=8,\
                                    majorValley=(majorValley, data2.Close[majorValley]),\
                                    majorPeak=(majorPeak, data2.Close[majorPeak]),\
                                    minorValley=(minorValley, data2.Close[minorValley]),\
                                    minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                    shortStart=(shortStart, data2.Close[shortStart]),\
                                    cycleList=cycleList,mode=modePred[start:],\
                                    signals=signalSets[is_period],chartTitle=ticker+' '+is_period,\
                                    savePath=chartSavePath+'_ODDS_'+is_period, debug=debug
                                    )
                else:
                    zz.plot_pivots(l=8,w=8,\
                                            startValley=(startValley, data2.Close[startValley]),\
                                            startPeak=(startPeak, data2.Close[startPeak]),\
                                            #minorValley=(minorValley, data2.Close[minorValley]),\
                                            #minorPeak=(minorPeak, data2.Close[minorPeak]),\
                                            shortStart=(shortStart, data2.Close[shortStart]),\
                                            cycleList=cycleList,mode=modePred[start:],\
                                            signals=signalSets[is_period],chartTitle=ticker+' '+is_period,\
                                            savePath=chartSavePath+'_ODDS_'+is_period, debug=debug
                                            )
    if showCharts:
        modes = mrClassifier(dataSet.Close[-(supportResistanceLB+validationSetLength):],\
                                            data.Close.shape[0],threshold=adfPvalue,\
                                            showPlot=debug, ticker=ticker, savePath=chartSavePath+'_MODE2')
        if debug:
            for x in signalSets:
                for algo in signalSets[x]:
                    signalSets[x][algo].to_csv('C:/users/hidemi/desktop/python/'+x+'_'+algo+'.csv')
            for x in DpsRankByMetricB:
                DpsRankByMetricB[x].to_csv('C:/users/hidemi/desktop/python/'+x+'.csv') 
            for x in DpsRankByMetricW:
                DpsRankByMetricW[x].to_csv('C:/users/hidemi/desktop/python/'+x+'.csv') 
            for x in finalDF:
                finalDF[x].to_csv('C:/users/hidemi/desktop/python/'+x+'.csv') 
            for x in signalDF:
                signalDF[x].to_csv('C:/users/hidemi/desktop/python/'+x+'.csv')
                
    for d in [DpsRankByMetricB, DpsRankByMetricW, finalDF]:
        for k, v in d.iteritems():
            signalDF[k]=v
            
    for is_period in signalSets:
        for k,v in signalSets[is_period].iteritems():
            signalDF[is_period+'_'+k]=v
            
    if showCharts:
        zz.plot_pivots(l=8,w=8,\
                            #startValley=(startValley, data2.Close[startValley]),\
                            #startPeak=(startPeak, data2.Close[startPeak]),\
                            #minorValley=(minorValley, data2.Close[minorValley]),\
                            #minorPeak=(minorPeak, data2.Close[minorPeak]),\
                            #shortStart=(shortStart, data2.Close[shortStart]),\
                            cycleList=cycleList,mode=modePred[start:],\
                            signals={useSignalsFrom:signalDF[useSignalsFrom]},chartTitle=ticker+\
                            ' '+useSignalsFrom+' metric '+curve+', bias '+bias[0]+\
                            ', Signal addAuxPairs '+str(addAuxPairs),\
                            savePath=chartSavePath+'_SIGNAL', debug=debug
                            )
                            
    sst=signalDF[useSignalsFrom].copy(deep=True)

    if useDPSsafef:
        sst['safef']=sst.dpsSafef
    else:
        sst['safef']=sst.nodpsSafef
        
    createSignalFile(version, version_, ticker, barSizeSetting, signalPath, sst, start_time, dataSet)
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes'
    return sst.iloc[-1]
    
if __name__ == "__main__":
    #system parameters
    version = 'v4'
    version_ = 'v4.3'
    asset = 'FX'
    #filterName = 'DF1'
    #data_type = 'ALL'
    barSizeSetting='30m'

    currencyPairs =   [
                    'NZDJPY',\
                    'CADJPY',\
                    'CHFJPY',\
                    'EURJPY',\
                    'GBPJPY',\
                    'AUDJPY',\
                    'USDJPY',\
                    'AUDUSD',\
                    'EURUSD',\
                    'EURAUD',\
                    'EURCAD',\
                    'EURNZD',\
                    'GBPUSD',\
                    'USDCAD',\
                    'USDCHF',\
                    'NZDUSD',
                    'EURCHF',\
                    'EURGBP',\
                    'AUDCAD',\
                    'AUDCHF',\
                    'AUDNZD',\
                    'GBPAUD',\
                    'GBPCAD',\
                    'GBPNZD',\
                    'GBPCHF',\
                    'CADCHF',\
                    'NZDCHF',\
                    'NZDCAD'
                    ]

    if len(sys.argv)==1:
        debug=True
        #gainAhead bias when 'choppy'
        #bias = ['gainAhead']
        bias = ['zigZag']
        #bias = ['gainAhead','zigZag']
        #bias=['sellHold']
        #bias=['buyHold']
        useSignalsFrom='tripleFiltered'
        #cycle mode->threshold=1.1
        #adfPvalue=1.1
        #trendmode -> threshold = -0.1
        #adfPvalue=-0.1
        #auto ->threshold = 0.2
        adfPvalue=0
        validationSetLength =48
        livePairs =  [
                        #'NZDJPY',\
                        #'CADJPY',\
                        #'CHFJPY',\
                        #'EURJPY',\
                        #'GBPJPY',\
                        #'AUDJPY',\
                        #'USDJPY',\
                        #'EURCHF',\
                        #'EURGBP',\
                        #'EURUSD',\
                        #'EURAUD',\
                        #'EURCAD',\
                        #'EURNZD',\
                        #'AUDUSD',\
                        #'GBPUSD',\
                        #'USDCAD',\
                        #'USDCHF',\
                        #'NZDUSD',
                        #'AUDCAD',\
                        #'AUDCHF',\
                        #'AUDNZD',\
                        #'GBPAUD',\
                        #'GBPCAD',\
                        #'GBPCHF',\
                        #'GBPNZD',\
                        #'NZDCHF',\
                        'NZDCAD',\
                        #'CADCHF'
                        ]
        ticker =livePairs[0]
        #dataPath =  'Z:/TSDP/data/from_IB/'
        dataPath = 'D:/ML-TSDP/data/from_IB/'
        signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
        #chartSavePath = None
        chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/'+version+'_'+ticker
        
        #adds auxilary pair features
        addAuxPairs = True
        
        #display params
        showCharts=False
        showFinalChartOnly=True
        showIndicators = False
        verbose=True
    else:
        debug=False
        livePairs=[sys.argv[1]]
        bias=[sys.argv[2]]
        adfPvalue=float(sys.argv[3])
        validationSetLength =int(sys.argv[4])
        useSignalsFrom=sys.argv[5]
        ticker =livePairs[0]
        #symbol=ticker[0:3]
        #currency=ticker[3:6]
        signalPath = './data/signals/'
        dataPath = './data/from_IB/'
        chartSavePath = './data/results/'+version+'_'+ticker
        
        #adds auxilary pair features
        addAuxPairs = False
        
        #display params
        showCharts=False
        showFinalChartOnly=True
        showIndicators = False
        verbose=False

    #Model Parameters
    supportResistanceLB = 90


    #for PCA/KBest
    nfeatures = 10
    #if major low/high most recent index. minDatapoints sets the minimum is period.
    minDatapoints = 3
    #set to 1 for live
    #system selection metric
    #metric = 'CAR25'
    metric = 'CAR25'
    #metric for signal
    metric2='netEquity'


    #robustness
    perturbData = False
    perturbDataPct = 0.0002


    #dps
    #save to signal file
    useDPSsafef=False
    #personal risk tolerance parameters
    PRT={}
    #ie goes to charts, car25 scoring (affected by commissions)
    PRT['initial_equity'] = 100000
    #fcst horizon(bars) for dps.  for training,  horizon is set to nrows.  for validation scoring nrows. 
    PRT['horizon'] = 50
    #safef set to dd95 where limit is met. e.g. for 50 bars, set saef to where 95% of the mc eq curves' maxDD <=0.01
    PRT['DD95_limit'] = 0.01
    PRT['tailRiskPct'] = 95
    #rounds safef and safef cannot go below this number. if set to None, no rounding
    PRT['minSafef'] =1
    #no dps safef
    PRT['nodpsSafef'] =2
    #dps max limit
    PRT['maxSafef'] = 2
    #safef=minSafef if CAR25 < threshold
    PRT['CAR25_threshold'] = 0
    #PRT['CAR25_threshold'] = -np.inf


    #needs 2x srlookback to prime indicators. 
    maxlb = supportResistanceLB*2
    maxReadLines = validationSetLength+maxlb
    initialEquity=PRT['initial_equity']
    nodpsSafef=PRT['nodpsSafef']
    #model selection
    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    fs_models = [
             ('PCA'+str(nfeatures),PCA(n_components=nfeatures)),\
             ('SelectKBest'+str(nfeatures),SelectKBest(f_classif, k=nfeatures))\
             ]
    short_models = [
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             ("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             #("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=2, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 #], voting='hard', weights=None)),
             #("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #        ], voting='soft', weights=None)),
             ]
    noAuxPairs_models = [
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             ("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=minDatapoints, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    ], voting='hard', weights=None)),
             #("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    #], voting='soft', weights=None)),
             ]
             
    auxPairs_models = [
             #("LR", LogisticRegression(class_weight={1:1})), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             #("VotingHard", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                    #], voting='hard', weights=None)),
             ("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=minDatapoints, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                     ], voting='soft', weights=None)),
             ]

    inner_zz_std =2
    outer_zz_std=4
    
    if len(sys.argv)==1:          
        for ticker in livePairs:
            runData = {'version':version, 'version_':version_,'asset':asset,'asset':asset,\
                            'barSizeSetting':barSizeSetting,'currencyPairs':currencyPairs,'debug':debug,'bias':bias,\
                            'adfPvalue':adfPvalue, 'validationSetLength':validationSetLength,'livePairs':livePairs,\
                            'ticker':ticker, 'dataPath':dataPath,'signalPath':signalPath,'chartSavePath':chartSavePath,\
                            'showCharts':showCharts, 'showFinalChartOnly':showFinalChartOnly,'showIndicators':showIndicators,\
                            'verbose':verbose, 'supportResistanceLB':supportResistanceLB,'nfeatures':nfeatures,\
                            'minDatapoints' :minDatapoints, 'metric':metric, 'metric2' :metric2,\
                            'addAuxPairs' :addAuxPairs, 'perturbData' :  perturbData, 'perturbDataPct' :perturbDataPct,\
                            'useDPSsafef' :useDPSsafef, 'PRT' :  PRT, 'maxlb' :maxlb,'maxReadLines':maxReadLines,\
                            'initialEquity':initialEquity,'nodpsSafef':nodpsSafef, 'fs_models':fs_models,\
                            'short_models':short_models,'noAuxPairs_models':noAuxPairs_models,'auxPairs_models':auxPairs_models,\
                            'outer_zz_std':outer_zz_std,'inner_zz_std':inner_zz_std}
                            
            signal = runv4(runData)
    else:
        if sys.argv[2] == 'debug':  
            debug = True
            logging.info( 'running debug mode...' )
            
        if sys.argv[1] == 'single':  
            while 1:
                start_time = time.time()
                logging.info( 'starting single thread mode for '+str(barSizeSetting)+' bars '+str(len(livePairs))+' pairs.')
                logging.info(str(livePairs) )
                get_bars(livePairs,barSizeSetting+'_')
                #time.sleep(100)
        elif sys.argv[1] == 'multi':
            runThreads()
        else:
            #print 'please specify single or multi, thanks.'
            sys.exit('please specify single or multi.')
    