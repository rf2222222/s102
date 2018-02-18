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
from display import sss_display_cmatrix, is_display_cmatrix, oos_display_cmatrix,\
                plot_learning_curve, is_display_cmatrix2, oos_display_cmatrix2,\
                update_report, displayRankedCharts
from transform import ratio, numberZeros, gainAhead
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

def CAR25_df_bars(signal_type, signals, signal_index, Close, **kwargs):
    verbose = kwargs.get('verbose', True)
    forecast_horizon=kwargs.get('minFcst',max(500,signal_index.shape[0]*3))
    DD95_limit = kwargs.get('DD95_limit',0.05)
    barSize = kwargs.get('barSize','30m')
    number_forecasts  = kwargs.get('number_forecasts',35)
    #minsafef sets lower bound on safef. 
    minSafef = kwargs.get('minSafef',None)
    safef= kwargs.get('startSafef',1)
    initial_equity=kwargs.get('initial_equity',100000.0)
    asset = kwargs.get('asset','FX')
    #barDict
    barDict = {
                    '1 min':60.0,
                    '10m':6.0,
                    '30m':2.0,
                    '1h':1.0
                    }
                    
    def IBcommission(tradeAmount, asset):
        commission = 2.0
        if asset == 'FX':
            return max(2.0, tradeAmount*2e-5)
        else:
            return commission
        
    holdBars = 1
    accuracy_tolerance = 0.005
    years_in_forecast = forecast_horizon/float(barDict[barSize]*24.0*365.0)
        
    #percentOfYearInMarket = number_long_signals /(years_in_study*252.0)
    #number_signals = index.shape[0]
    number_trades = forecast_horizon / holdBars
    numBars = number_trades*holdBars
    account_balance = np.zeros(numBars+1, dtype=float) 
    max_IT_DD = np.zeros(numBars+1, dtype=float)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(numBars+1, dtype=float)     # Maximum Intra-Trade equity
    FC_max_IT_DD = np.zeros(number_forecasts, dtype=float) # Max intra-trade drawdown
    FC_tr_eq = np.zeros(number_forecasts, dtype=float)     # Trade equity (TWR)
    FC_trades = np.zeros(number_forecasts, dtype=float)     
    FC_sortino = np.zeros(number_forecasts, dtype=float)     
    FC_sharpe = np.zeros(number_forecasts, dtype=float)

    # start loop
    done = False    
    while not done:
        done = True
        #print 'Using safef: %.3f ' % safef,
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
            #print "forecast ",i_forecast, " of ", number_forecasts
            #if i_forecast >0:
             #   print i_forecast-1, 'max_IT_DD', max_IT_DD[i_forecast]
            #wait = raw_input("PRESS ENTER TO CONTINUE.")
            #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
            trades = 0
            lastPosition = 0
            #  for each trade
            #for i_trade in range(0,number_trades):
            for i_trade in range(0,number_trades):
                #print 'day', i_trade, 'of',number_trades
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                index = random.choice(range(0,len(signal_index)-1))
                TRADE = signals[index]
                
                
                if TRADE > 0:
                    direction = 'LONG'
                else:
                    direction = 'SHORT'
                #print direction, safef
                if TRADE: #<0 not toxic
                    entry_index = signal_index[index]
                    #rint entry_index, TRADE
                    #  Process the trade, day by day
                    for i_day_in_trade in range(0,holdBars+1):
                        if i_day_in_trade==0: #day 0
                            #  Things that happen immediately 
                            #  after the close of the signal day
                            #  Initialize for the trade
                            entry_price = Close.ix[entry_index]
                            #print entry_price

                            #print account_balance[i_day], safef, buy_price
                            number_shares = account_balance[i_day] * \
                                            safef / entry_price
                            share_dollars = number_shares * entry_price
                            cash = account_balance[i_day] - \
                                   share_dollars
                            #print '155', "buy price", number_shares, "num_shares", share_dollars, "share $", cash, "cash"

                        else: # day n
                            #print 'iday in trade', i_day_in_trade, 'iday', i_day, 'num trades', number_trades
                            #  Things that change during a 
                            #  day the trade is held
                            i_day = i_day + 1
                            j = entry_index + i_day_in_trade
                            #print Close.ix[j], entry_price
                            #  Drawdown for the trade

                            if direction == 'LONG':
                                profit = number_shares * (Close.ix[j] - entry_price)
                            else:                            
                                profit = number_shares * (entry_price - Close.ix[j])

                            if i_day_in_trade==holdBars: # last day of forecast
                                #  Exit at the close
                                exit_price = Close.ix[j]
                            addCommission = 0
                            #min safef specifies safef as a whole number.
                            if minSafef is not None:    
                                if TRADE*round(safef) != lastPosition:
                                    trades += 1
                                    #print trades, lastPosition, TRADE*round(safef)
                                    lastPosition = TRADE*round(safef)
                                    #print trades, lastPosition, TRADE*round(safef)
                                    addCommission = -IBcommission(share_dollars+profit, asset)
                            else:
                                if TRADE*safef != lastPosition:
                                    trades += 1
                                    #print trades, lastPosition, TRADE*safef
                                    lastPosition = TRADE*safef
                                    #print trades, lastPosition, TRADE*safef
                                    addCommission = -IBcommission(share_dollars+profit, asset)
                            #print addCommission
                            MTM_equity = cash + share_dollars + profit +addCommission
                            IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                    / max_IT_Eq[i_day-1]
                            max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                    IT_DD)
                            max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                    MTM_equity)
                            account_balance[i_day] = MTM_equity
                            #print '175accountbal', account_balance[i_day],'profit', profit, 'itdd', IT_DD,'max_IT_DD[i_day]',max_IT_DD[i_day], 'max_ITEq', max_IT_Eq[i_day]
                                    
                            # Check for end of forecast
                            #print 'i_day', i_day,'numBars',numBars
                            if i_day >= numBars:
                                #print '182##############ENDSAVE i_day', i_day, 'numBars', numBars
                                FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                                FC_tr_eq[i_forecast] = MTM_equity
                                FC_trades[i_forecast] = trades
                                FC_sortino[i_forecast] = ratio(account_balance).sortino()
                                FC_sharpe[i_forecast] = ratio(account_balance).sharpe()
                        #print '207maxitdd', max_IT_DD[i_day]
                                    #print '186maxitdd', max_IT_DD[i_day]
                                    

                else: # no trade
                    #print '189iday', i_day, 'num days', numBars, 'num trades', number_trades
                    MTM_equity = account_balance[i_day]
                    i_day = i_day + 1
                    IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                            / max_IT_Eq[i_day-1]
                    max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                            IT_DD)
                    max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                            MTM_equity)
                    account_balance[i_day] = MTM_equity
                    #print '199no_trade ', 'mtm', MTM_equity, 'itdd', IT_DD, 'max_IT_DD', max_IT_DD[i_day], 'max_IT_Eq', max_IT_Eq[i_day]
                    #print i_day, numBars
                    # Check for end of forecast
                    if i_day >= numBars:
                        #print '##############ENDSAVE i_day', i_day, 'numBars', numBars
                        FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                        FC_tr_eq[i_forecast] = MTM_equity
                        FC_trades[i_forecast] = trades
                        FC_sortino[i_forecast] = ratio(account_balance).sortino()
                        FC_sharpe[i_forecast] = ratio(account_balance).sharpe()
                        #print '207maxitdd', max_IT_DD[i_day]

        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile 
        #print '211maxdd ', FC_max_IT_DD       
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)    
        #print 'DD_95',DD_95
        
        #stop at minimum safef
        if minSafef is not None:
            if safef > minSafef:
                #adjust safef
                if (abs(DD95_limit - DD_95) < accuracy_tolerance):
                    #print '  214 DD95: %.3f ' % DD_95, 'DD95_limit',DD95_limit,"Close enough" 
                    done = True
                elif DD_95 == 0: #no drawdown
                    #print 'dd95 =0'
                    safef =  float('inf')
                    done == True
                elif DD_95 == 1: #max loss
                    #print 'dd95=1'
                    safef = 0
                    done == True 
                else:
                    #print '  DD95: %.3f ' % DD_95, "DD95_limit",DD95_limit," Adjust safef from " , safef,
                    safef = safef * DD95_limit / DD_95
                    if safef < minSafef:
                        safef = minSafef
                    #print 'to ', safef, 'minsafef',minSafef
                    done = False   
            else:
                #stop when safef<min safef
                done==True
        else:
            if (abs(DD95_limit - DD_95) < accuracy_tolerance):
                #print '  214 DD95: %.3f ' % DD_95, "Close enough" 
                done = True
            elif DD_95 == 0: #no drawdown
                #print 'dd95 =0'
                safef =  float('inf')
                done == True
            elif DD_95 == 1: #max loss
                #print 'dd95=1'
                safef = 0
                done == True 
            else:
                #print '  DD95: %.3f ' % DD_95, "DD95_limit",DD95_limit," Adjust safef from " , safef,
                safef = safef * DD95_limit / DD_95
                done = False
            
    #  Report
    SIG = signal_type
    YIF =  forecast_horizon
    TPY = FC_trades.mean()/(forecast_horizon)
    IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
    IT_DD_100 = max(FC_max_IT_DD)
    SOR25 = stats.scoreatpercentile(FC_sortino,25)
    SHA25 = stats.scoreatpercentile(FC_sharpe,25)
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
    TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
    CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
    CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
    CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
        
    if verbose:
        print '\nSignal: ', SIG
        print 'Fcst Horizon (%s Bars): %i' % (barSize, YIF) ,
        #print ' TotalTrades: %.0f ' % FC_trades.mean(), 
        print ' Avg.Trades/Bar: %.1f' % TPY
        print 'DD95:  %.3f ' % IT_DD_95,
        print 'DD100: %.3f ' %  IT_DD_100,
        print 'SORTINO25: %.3f ' %  SOR25,
        print 'SHARPE25: %.3f ' % SHA25
        print 'SAFEf: %.3f ' % safef,

        print 'CAR25: %.3f ' % CAR_25,
        print 'CAR50: %.3f ' % CAR_50,
        print 'CAR75: %.3f ' % CAR_75
    metrics = {'C25sig':SIG, 'safef':safef, 'CAR25':CAR_25, 'CAR50':CAR_50, 'CAR75':CAR_75,\
                'DD95':IT_DD_95, 'DD100':IT_DD_100, 'SOR25':SOR25, 'SHA25':SHA25, 'YIF':YIF, 'TPY':TPY}

    return metrics
    
def wf_classify_validate2(unfilteredData, dataSet, m, model_metrics, \
                           metaData, **kwargs):
    showPDFCDF= kwargs.get('showPDFCDF',True)
    showLearningCurve=kwargs.get('showLearningCurve',False)
    longMemory=kwargs.get('longMemory',False)
    verbose=kwargs.get('verbose',True)
    PDFCDFsavePath=kwargs.get('PDFCDFsavePath',None)
    PDFCDFfilename=kwargs.get('PDFCDFfilename',None)
    
    close = unfilteredData.reset_index().Close
    #fill in the prior index. need this for the car25 calc uses the close index
    unfilteredData['prior_index'] = pd.concat([dataSet.prior_index, unfilteredData.Close],axis=1,join='outer').prior_index.interpolate(method='linear').dropna()
    ticker = metaData['ticker']
    data_type = metaData['data_type']
    iterations = metaData['iters']
    testFinalYear= metaData['t_end']
    validationFirstYear=metaData['v_start']
    validationFinalYear=metaData['v_end']
    wfStep=metaData['wf_step']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    tox_adj_proportion = metaData['tox_adj']
    feature_selection = metaData['FS']
    wf_is_period = metaData['wf_is_period']
    ddTolerance = metaData['DD95_limit']
    forecastHorizon = metaData['horizon']
    barSize = metaData['barSizeSetting']
    initial_equity=metaData['initial_equity']
    
    #CAR25 for zigzag
    if signal[:2]=='ZZ':
        zz_step = [float(x) for x in signal.split('_')[1].split(',')]

    #create signals
    if signal != 'GA1' or signal != 'gainAhead':
        #gainAhead with lookforward
        if signal[:2] == 'GA':
            #wfStep = int(signal[2:])
            #ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            ga_start = dataSet.iloc[0].prior_index
            ga = gainAhead(close.ix[ga_start:],wfStep)
            dataSet.signal = np.array([-1 if x<0 else 1 for x in ga])
            
        #zigzag
        if signal[:2] == 'ZZ':
            zz_end = dataSet.iloc[-1].prior_index
            dataSet.signal = pd.Series(zg(close.ix[:zz_end].values, zz_step[0], \
                            zz_step[1]).pivots_to_modes()[-dataSet.shape[0]:]).shift(-1).fillna(0).values
                            
        #buyHold                    
        if signal[:2] == 'BH':
            dataSet.signal = pd.Series(data=1,index=dataSet.index)
        
        #sellHold
        if signal[:2] == 'SH':
            dataSet.signal = pd.Series(data=-1,index=dataSet.index)
            
    metaData['wf_step']=wfStep


    if 'filter' in metaData:
        filterName = metaData['filter']
    else:
        filterName = 'OOS_V'

    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']

    #check
    nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
    if wf_is_period > nrows_is:
        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        print 'Adjusting to', nrows_is, 'rows..'
        wf_is_period = nrows_is

    mmData = dataSet.ix[:testFinalYear].dropna()[-wf_is_period:]
    mmData_adj = adjustDataProportion(mmData, tox_adj_proportion)  #drop last row for hold days =1
    mmData_v = pd.concat([mmData_adj,dataSet.ix[validationFirstYear:validationFinalYear].dropna()], axis=0).reset_index()

    nrows_is = mmData.shape[0]
    nrows_oos = mmData_v.shape[0]-nrows_is
        
    metaData['rows'] = nrows_is

    #nrows = mmData_adj.shape[0]
    datay_signal = mmData_v[['signal', 'prior_index']]
    datay_gainAhead = mmData_v.gainAhead

    dataX = mmData_v.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    
    feature_names = []
    if verbose == True:
        print '\nTotal %i features: ' % cols
        
    for i,x in enumerate(dataX.columns):
        if verbose == True:
            print i,x+',',
        feature_names = feature_names+[x]
        
    if feature_selection != 'None':
        if nfeatures > cols:
            print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
            print 'Adjusting to', cols, 'features..'
            nfeatures = cols  
        metaData['cols']=nfeatures
    
            
    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay_signal.signal)
    dX = np.zeros_like(dataX)

    dy = datay_signal.signal.values
    dX = dataX.values
    
    if signal[:2] == 'BH' or signal[:2] == 'SH':
        cm_test_index = range(wf_is_period,mmData_v.shape[0])
        
        if signal[:2] == 'BH':
            cm_y_test = np.ones(len(cm_test_index))
            cm_y_pred_oos = np.ones(len(cm_test_index))
        else:
            cm_y_test = -np.ones(len(cm_test_index))
            cm_y_pred_oos = -np.ones(len(cm_test_index))
            
    else:
        #for m in models:
        if verbose == True:
            print '\n\nNew WF train/predict loop for', m[1]
            print "\nStarting Walk Forward run on", metaData['data_type'], "data..."
            if feature_selection == 'Univariate':
                print "Using top %i %s features" % (nfeatures, feature_selection)
            else:
                print "Using features selection: %s " % feature_selection
            print 'Signal type:', signal
            if longMemory == False:
                print "%i rows in sample, %i rows out of sample, forecasting %i bar(s) ahead.." % (nrows_is, nrows_oos,wfStep)
            else:
                print "long memory starting with %i rows in sample, %i rows out of sample, forecasting %i bar(s) ahead.." % (nrows_is, nrows_oos,wfStep)
            #cm_y_train = np.array([])
        cm_y_test = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        leftoverIndex = nrows_oos%wfStep
        
        #reverse index to equate the wf tests of different periods, count backwards from the end
        wfIndex = range(nrows_oos-wfStep,-wfStep,-wfStep)
        tt_index =[]
        for i in wfIndex:
            #last wf index adjust the test index, else step
            if leftoverIndex > 0 and i == wfIndex[-1]:
                train_index = range(0,wf_is_period)        
                test_index = range(wf_is_period,wf_is_period+leftoverIndex)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
            else:
                if longMemory == True:
                    train_index = range(0,wf_is_period+i)
                else:
                    train_index = range(i,wf_is_period+i)
                #the last wfStep indexes are untrained.
                test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
        #c=0
        #zz_begin = mmData_v.prior_index.iloc[0]
        for train_index,test_index in tt_index:
            #c+=1
            X_train, X_test = dX[train_index], dX[test_index]
            y_train, y_test = dy[train_index], dy[test_index]
            
            #create zigzag signals
            
            #ending at test_index so dont need to shift labels
            #if signal[:3] != 'GA1':
            #    if signal[:2] == 'GA':
            #        lookforward = int(signal_types[2][2:])
            #        ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            #        #ga_start = dataSet.iloc[0].prior_index
            #        ga = gainAhead(close.ix[ga_start:],lookforward)
            #        y_train = np.array([-1 if x<0 else 1 for x in ga])[:len(train_index)]
                    
            if signal[:2] == 'ZZ':
                zz_end = mmData_v.iloc[test_index].prior_index.iloc[len(test_index)-1]
                y_train = zg(close.ix[:zz_end].values, zz_step[0], \
                                zz_step[1]).pivots_to_modes()[-len(train_index):]
                
            #check if there are no intersections
            intersect = np.intersect1d(datay_signal.reset_index().iloc[test_index]['index'].values,\
                        datay_signal.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
            if len(mmData_v.index[-wfStep:].intersection(train_index)) == 0:
                #print 'training', X_train.shape
                if feature_selection != 'None':
                    #print 'using feature selection',feature_selection
                    if feature_selection == 'RFECV':
                        #Recursive feature elimination with cross-validation: 
                        #A recursive feature elimination example with automatic tuning of the
                        #number of features selected with cross-validation.
                        rfe = RFECV(estimator=RFE_estimator, step=1)
                        rfe.fit(X_train, y_train)
                        #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                        featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                        #print 'Top %i RFECV features' % len(featureRank)
                        #print featureRank    
                        metaData['featureRank'] = str(featureRank)
                        X_train = rfe.transform(X_train)
                        X_test = rfe.transform(X_test)
                    else:
                        #Univariate feature selection
                        skb = SelectKBest(f_regression, k=nfeatures)
                        skb.fit(X_train, y_train)
                        #dX_all = np.vstack((X_train.values, X_test.values))
                        #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                        #dX_v_rfe = X_new[dX_t.shape[0]:]
                        X_train = skb.transform(X_train)
                        X_test = skb.transform(X_test)
                        featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                        metaData['featureRank'] = str(featureRank)
                        #print 'Top %i univariate features' % len(featureRank)
                        #print featureRank

                #  fit the model to the in-sample data
                m[1].fit(X_train, y_train)
                if verbose:
                    print m[0], signal, X_train.shape,

                #trained_models[m[0]] = pickle.dumps(m[1])
                            
                #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
                y_pred_oos = m[1].predict(X_test)
                #print y_pred_oos.shape

                if m[0][:2] == 'GA':
                    print featureRank
                    print '\nProgram:', m[1]._program
                    #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
                
                #cm_y_train = np.concatenate([cm_y_train,y_train])
                cm_y_test = np.concatenate([cm_y_test,y_test])
                #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
                cm_y_pred_oos = np.concatenate([cm_y_pred_oos,y_pred_oos])
                #cm_train_index = np.concatenate([cm_train_index,train_index])
                cm_test_index = np.concatenate([cm_test_index,test_index])
            

    #create signals 1 and -1
    #cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
    #cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
    
    #gives errors when 100% accuracy for binary classification
    #if confusion_matrix(cm_y_test[:-1], cm_y_pred_oos[:-1]).shape == (1,1):
    #    print  m[0], ticker,validationFirstYear, validationFinalYear, iterations, signal
    #    print 'Accuracy 100% for', cm_y_test[:-1].shape[0], 'rows'
    #else:
    if verbose == True:
        #print cm_y_test[:-wfStep].shape[0], cm_y_pred_oos[:-wfStep].shape[0]
        if cm_y_test[:-wfStep].shape[0]>0:
        #if wfStep>1:
            oos_display_cmatrix(cm_y_test[:-wfStep], cm_y_pred_oos[:-wfStep], m[0],\
                ticker,validationFirstYear, dataSet.index[-wfStep], iterations, signal)
        #else:
        #    oos_display_cmatrix(cm_y_test[:-1], cm_y_pred_oos[:-1], m[0],\
        #            ticker,validationFirstYear, validationFinalYear, iterations, signal)
    #if data is filtered so need to fill in the holes. signal = 0 for days that filtered
    st_oos_filt= pd.DataFrame()
    st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
    st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
            
    #compute car, show matrix if data is filtered
    if data_type != 'ALL':
        
        prior_index_filt = pd.concat([st_oos_filt,unfilteredData.prior_index], axis=1,\
                            join='inner').prior_index.values.astype(int)
        #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
        if verbose == True or showPDFCDF == True or (PDFCDFsavePath != None and PDFCDFfilename != None):
            #print 'Metrics for filtered Validation Datapoints'
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                    ticker, validationFirstYear, validationFinalYear, iterations, metaData['filter'],showPDFCDF=showPDFCDF,\
                    savePath=PDFCDFsavePath, filename=PDFCDFfilename,verbose=verbose)
        CAR25_oos = CAR25_df_bars(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                close, DD95_limit =ddTolerance, verbose=verbose,barSize=barSize, minSafef=1,\
                                initial_equity=initial_equity)
        #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
        #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
                                
    #add column prior index and gA.  if there are holes, nan values in signals
    st_oos_filt = pd.concat([st_oos_filt,unfilteredData.gainAhead,unfilteredData.prior_index],\
                                axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
    #fills nan with zeros
    st_oos_filt['signals'].fillna(0, inplace=True)
    
    #fill zeros with opposite of input signal, if there are zeros. to return full data
    cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,metaData['input_signal']*-1,\
                                                                st_oos_filt['signals'].values)
    cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
    #datay_gainAhead and cmatrix_test_index have the same index
    datay_gainAhead = st_oos_filt.gainAhead
    cmatrix_test_index = st_oos_filt.reset_index().index

    #plot learning curve, knn insufficient neighbors
    if showLearningCurve:
        try:
            plot_learning_curve(m[1], m[0], X_train,y_train_ga, scoring='r2')        
        except:
            pass

    
    #compute car, show matrix for all data is unfiltered
    if data_type == 'ALL':           
        if verbose == True or showPDFCDF == True or (PDFCDFsavePath != None and PDFCDFfilename != None):
            #print 'Metrics for All Validation Datapoints'
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cmatrix_test_index, m[1], ticker,\
                                validationFirstYear, validationFinalYear, iterations, 'Long>0',showPDFCDF=showPDFCDF,\
                                 savePath=PDFCDFsavePath, filename=PDFCDFfilename,verbose=verbose)
        #minfraction set to 1 because no odd lots. 
        CAR25_oos = CAR25_df_bars(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                close, DD95_limit =ddTolerance, verbose=verbose, barSize=barSize, minSafef=1,\
                                initial_equity=initial_equity)
        #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
         #                       close, 'LONG', 1)
        #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
         #                       close, 'SHORT', -1)
    #update model metrics
    #metaData['signal'] = 'LONG 1'
    model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                            cmatrix_test_index, m, metaData,CAR25_oos)
    #metaData['signal'] = 'SHORT -1'
    #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
    #                       cmatrix_test_index, m, metaData,CAR25_Sn1_oos)
    return model_metrics, st_oos_filt, m[1]
    
def wf_classify_validate(unfilteredData, dataSet, models, model_metrics, wf_is_period, \
                           metaData, PRT, showPDFCDF=True, showLearningCurve=False, longMemory=False):
    close = unfilteredData.reset_index().Close
    #fill in the prior index. need this for the car25 calc uses the close index
    unfilteredData['prior_index'] = pd.concat([dataSet.prior_index, unfilteredData.Close],axis=1,join='outer').prior_index.interpolate(method='linear').dropna()
    ticker = metaData['ticker']
    data_type = metaData['data_type']
    iterations = metaData['iters']
    testFinalYear= metaData['t_end']
    validationFirstYear=metaData['v_start']
    validationFinalYear=metaData['v_end']
    wfStep=metaData['wf_step']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    tox_adj_proportion = metaData['tox_adj']
    feature_selection = metaData['FS']
    ddTolerance = PRT['DD95_limit']
    forecastHorizon = PRT['horizon']
    
    #CAR25 for zigzag
    if signal[:2]=='ZZ':
        zz_step = [float(x) for x in signal.split('_')[1].split(',')]

    #create signals
    if signal != 'GA1' or signal != 'gainAhead':
        if signal[:2] == 'GA':
            #wfStep = int(signal[2:])
            #ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            ga_start = dataSet.iloc[0].prior_index
            ga = gainAhead(close.ix[ga_start:],wfStep)
            dataSet.signal = np.array([-1 if x<0 else 1 for x in ga])
            
        if signal[:2] == 'ZZ':
            zz_end = dataSet.iloc[-1].prior_index
            dataSet.signal = pd.Series(zg(close.ix[:zz_end].values, zz_step[0], \
                            zz_step[1]).pivots_to_modes()[-dataSet.shape[0]:]).shift(-1).fillna(0).values

    else:
        wfStep=1
    metaData['wf_step']=wfStep
                            



    #adjust personal risk tolerance for available data
    #assuming dd increases with sqrt of time
    #ddTolerance = ddTolerance/math.sqrt(float(forecastHorizon)/float(wf_is_period))
    #forecastHorizon = wf_is_period #need a bit more than 2x trades of the fcst horizon

    if 'filter' in metaData:
        filterName = metaData['filter']
    else:
        filterName = 'OOS_V'

    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']

    #check
    nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
    if wf_is_period > nrows_is:
        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        print 'Adjusting to', nrows_is, 'rows..'
        wf_is_period = nrows_is

    mmData = dataSet.ix[:testFinalYear].dropna()[-wf_is_period:]
    mmData_adj = adjustDataProportion(mmData, tox_adj_proportion)  #drop last row for hold days =1
    mmData_v = pd.concat([mmData_adj,dataSet.ix[validationFirstYear:validationFinalYear].dropna()], axis=0).reset_index()

    nrows_is = mmData.shape[0]
    nrows_oos = mmData_v.shape[0]-nrows_is
        
    metaData['rows'] = nrows_is

    #nrows = mmData_adj.shape[0]
    datay_signal = mmData_v[['signal', 'prior_index']]
    datay_gainAhead = mmData_v.gainAhead

    dataX = mmData_v.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    feature_names = []
    print '\nTotal %i features: ' % cols
    for i,x in enumerate(dataX.columns):
        print i,x+',',
        feature_names = feature_names+[x]
    if nfeatures > cols:
        print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
        print 'Adjusting to', cols, 'features..'
        nfeatures = cols  
            
    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay_signal.signal)
    dX = np.zeros_like(dataX)

    dy = datay_signal.signal.values
    dX = dataX.values
    for m in models:
        print '\n\nNew WF train/predict loop for', m[1]
        print "\nStarting Walk Forward run on", metaData['data_type'], "data..."
        if feature_selection == 'Univariate':
            print "Using top %i %s features" % (nfeatures, feature_selection)
        else:
            print "Using %s features" % feature_selection
        if longMemory == False:
            print "%i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        else:
            print "long memory starting with %i rows in sample, %i rows out of sample, forecasting %i bar(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        #cm_y_train = np.array([])
        cm_y_test = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        leftoverIndex = nrows_oos%wfStep
        
        #reverse index to equate the wf tests of different periods, count backwards from the end
        wfIndex = range(nrows_oos-wfStep,-wfStep,-wfStep)
        tt_index =[]
        for i in wfIndex:
            #last wf index adjust the test index, else step
            if leftoverIndex > 0 and i == wfIndex[-1]:
                train_index = range(0,wf_is_period)        
                test_index = range(wf_is_period,wf_is_period+leftoverIndex)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
            else:
                if longMemory == True:
                    train_index = range(0,wf_is_period+i)
                else:
                    train_index = range(i,wf_is_period+i)
                #the last wfStep indexes are untrained.
                test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
        #c=0
        #zz_begin = mmData_v.prior_index.iloc[0]
        for train_index,test_index in tt_index:
            #c+=1
            X_train, X_test = dX[train_index], dX[test_index]
            y_train, y_test = dy[train_index], dy[test_index]
            
            #create zigzag signals
            
            #ending at test_index so dont need to shift labels
            #if signal[:3] != 'GA1':
            #    if signal[:2] == 'GA':
            #        lookforward = int(signal_types[2][2:])
            #        ga_start = mmData_v.iloc[train_index].prior_index.iloc[0]
            #        #ga_start = dataSet.iloc[0].prior_index
            #        ga = gainAhead(close.ix[ga_start:],lookforward)
            #        y_train = np.array([-1 if x<0 else 1 for x in ga])[:len(train_index)]
                    
            if signal[:2] == 'ZZ':
                zz_end = mmData_v.iloc[test_index].prior_index.iloc[len(test_index)-1]
                y_train = zg(close.ix[:zz_end].values, zz_step[0], \
                                zz_step[1]).pivots_to_modes()[-len(train_index):]
                
            #zz_signals = pd.DataFrame()
            #print 'Creating Signal labels..',
            #for i in zz_steps:
            #    for j in zz_steps:
            #        label = 'ZZ '+str(i) + ',-' + str(j)
            #        print label,


            #CAR25_list = []
            #for sig in zz_signals.columns:
            #    CAR25 = CAR25_df_min(sig,zz_signals[sig], close.ix[:zz_end].index,\
            #                        close.ix[:zz_end], minFcst=forecastHorizon, DD95_limit =ddTolerance)                    
            #    CAR25_list.append(CAR25)

            #CAR25_MAX = maxCAR25(CAR25_list) 
            #print '\nBest Signal Labels Found.', CAR25_MAX['C25sig']
            #signal = CAR25_MAX['C25sig']
            #y_train = zz_signals[CAR25_MAX['C25sig']].values[-len(train_index):]

            #print mmData_v.dates.iloc[test_index[-1]],
            #print 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
            #        'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
            #print train_index, test_index
            #check if there are no intersections
            intersect = np.intersect1d(datay_signal.reset_index().iloc[test_index]['index'].values,\
                        datay_signal.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
            if len(mmData_v.index[-wfStep:].intersection(train_index)) == 0:
                #print 'training', X_train.shape
                if feature_selection == 'RFECV':
                    #Recursive feature elimination with cross-validation: 
                    #A recursive feature elimination example with automatic tuning of the
                    #number of features selected with cross-validation.
                    rfe = RFECV(estimator=RFE_estimator, step=1)
                    rfe.fit(X_train, y_train)
                    #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                    featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                    print 'Top %i RFECV features' % len(featureRank)
                    print featureRank    
                    metaData['featureRank'] = str(featureRank)
                    X_train = rfe.transform(X_train)
                    X_test = rfe.transform(X_test)
                else:
                    #Univariate feature selection
                    skb = SelectKBest(f_regression, k=nfeatures)
                    skb.fit(X_train, y_train)
                    #dX_all = np.vstack((X_train.values, X_test.values))
                    #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                    #dX_v_rfe = X_new[dX_t.shape[0]:]
                    X_train = skb.transform(X_train)
                    X_test = skb.transform(X_test)
                    featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                    metaData['featureRank'] = str(featureRank)
                    #print 'Top %i univariate features' % len(featureRank)
                    #print featureRank

                #  fit the model to the in-sample data
                m[1].fit(X_train, y_train)


                #trained_models[m[0]] = pickle.dumps(m[1])
                            
                #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
                y_pred_oos = m[1].predict(X_test)

                if m[0][:2] == 'GA':
                    print featureRank
                    print '\nProgram:', m[1]._program
                    #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
                
                #cm_y_train = np.concatenate([cm_y_train,y_train])
                cm_y_test = np.concatenate([cm_y_test,y_test])
                #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
                cm_y_pred_oos = np.concatenate([cm_y_pred_oos,y_pred_oos])
                #cm_train_index = np.concatenate([cm_train_index,train_index])
                cm_test_index = np.concatenate([cm_test_index,test_index])
            

        #create signals 1 and -1
        #cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
        #cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
        
        #gives errors when 100% accuracy for binary classification
        #if confusion_matrix(cm_y_test[:-1], cm_y_pred_oos[:-1]).shape == (1,1):
        #    print  m[0], ticker,validationFirstYear, validationFinalYear, iterations, signal
        #    print 'Accuracy 100% for', cm_y_test[:-1].shape[0], 'rows'
        #else:
        if wfStep>1:
            oos_display_cmatrix(cm_y_test[:-wfStep], cm_y_pred_oos[:-wfStep], m[0],\
                ticker,validationFirstYear, dataSet.index[-wfStep], iterations, signal)
        else:
            oos_display_cmatrix(cm_y_test[:-1], cm_y_pred_oos[:-1], m[0],\
                    ticker,validationFirstYear, validationFinalYear, iterations, signal)
        #if data is filtered so need to fill in the holes. signal = 0 for days that filtered
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
                
        #compute car, show matrix if data is filtered
        if data_type != 'ALL':
            print 'Metrics for filtered Validation Datapoints'
            prior_index_filt = pd.concat([st_oos_filt,unfilteredData.prior_index], axis=1,\
                                join='inner').prior_index.values.astype(int)
            #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                    ticker, validationFirstYear, validationFinalYear, iterations, metaData['filter'],showPDFCDF)
            CAR25_oos = CAR25_df_min(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=forecastHorizon, DD95_limit =ddTolerance)
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
                                    
        #add column prior index and gA.  if there are holes, nan values in signals
        st_oos_filt = pd.concat([st_oos_filt,unfilteredData.gainAhead,unfilteredData.prior_index],\
                                    axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
        #fills nan with zeros
        st_oos_filt['signals'].fillna(0, inplace=True)
        
        #fill zeros with opposite of input signal, if there are zeros. to return full data
        cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,metaData['input_signal']*-1,\
                                                                    st_oos_filt['signals'].values)
        cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
        #datay_gainAhead and cmatrix_test_index have the same index
        datay_gainAhead = st_oos_filt.gainAhead
        cmatrix_test_index = st_oos_filt.reset_index().index

        #plot learning curve, knn insufficient neighbors
        if showLearningCurve:
            try:
                plot_learning_curve(m[1], m[0], X_train,y_train_ga, scoring='r2')        
            except:
                pass
            
        #plot out-of-sample data
        #if showPDFCDF:
        #    plt.figure()
        #    coef, b = np.polyfit(cm_y_pred_oos_ga, cm_y_test_ga, 1)
        #    plt.title('Out-of-Sample')
        #    plt.ylabel('gainAhead')
        #    plt.xlabel('ypred gainAhead')
        #    plt.plot(cm_y_pred_oos_ga, cm_y_test_ga, '.')
        #    plt.plot(cm_y_pred_oos_ga, coef*cm_y_pred_oos_ga + b, '-')
        #    plt.show()
        
        #compute car, show matrix for all data is unfiltered
        if data_type == 'ALL':
            print 'Metrics for All Validation Datapoints'
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cmatrix_test_index, m[1], ticker,\
                                validationFirstYear, validationFinalYear, iterations, 'Long>0',showPDFCDF)
            CAR25_oos = CAR25_df_min(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=forecastHorizon, DD95_limit =ddTolerance)
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'SHORT', -1)
        #update model metrics
        #metaData['signal'] = 'LONG 1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                                cmatrix_test_index, m, metaData,CAR25_oos)
        #metaData['signal'] = 'SHORT -1'
        #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
        #                       cmatrix_test_index, m, metaData,CAR25_Sn1_oos)
    return model_metrics, st_oos_filt

def wf_regress_validate2(unfilteredData, dataSet, models, model_metrics, wf_is_period, \
                           metaData, PRT, showPDFCDF=True, showLearningCurve=False, longMemory=False):
    close = unfilteredData.reset_index().Close
    #fill in the prior index. need this for the car25 calc uses the close index
    unfilteredData['prior_index'] = pd.concat([dataSet.prior_index, unfilteredData.Close],axis=1,join='outer').prior_index.interpolate(method='linear').dropna()
    ticker = metaData['ticker']
    data_type = metaData['data_type']
    iterations = metaData['iters']
    testFinalYear= metaData['t_end']
    validationFirstYear=metaData['v_start']
    validationFinalYear=metaData['v_end']
    wfStep=metaData['wf_step']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    tox_adj_proportion = metaData['tox_adj']
    feature_selection = metaData['FS']
    RFE_estimator = metaData['rfe_model'][1]
    metaData['rfe_model'] = metaData['rfe_model'][0]
    if 'filter' in metaData:
        filterName = metaData['filter']
    else:
        filterName = 'OOS_V'
    
    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']

    #check
    nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
    if wf_is_period > nrows_is:
        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        print 'Adjusting to', nrows_is, 'rows..'
        wf_is_period = nrows_is
    
    mmData = dataSet.ix[:testFinalYear].dropna()[-wf_is_period:]
    mmData_adj = adjustDataProportion(mmData, tox_adj_proportion)  #drop last row for hold days =1
    mmData_v = pd.concat([mmData_adj,dataSet.ix[validationFirstYear:validationFinalYear].dropna()], axis=0).reset_index()
    
    nrows_is = mmData.shape[0]
    nrows_oos = mmData_v.shape[0]-nrows_is
        
    metaData['rows'] = nrows_is
    
    #nrows = mmData_adj.shape[0]
    datay_signal = mmData_v[['signal', 'prior_index']]
    datay_gainAhead = mmData_v.gainAhead
    
    dataX = mmData_v.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    feature_names = []
    print '\nTotal %i features: ' % cols
    for i,x in enumerate(dataX.columns):
        print i,x+',',
        feature_names = feature_names+[x]
    if nfeatures > cols:
        print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
        print 'Adjusting to', cols, 'features..'
        nfeatures = cols  
            
    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay_gainAhead)
    dX = np.zeros_like(dataX)
    
    dy = datay_gainAhead.values
    dX = dataX.values
    for m in models:
        print '\n\nNew WF train/predict loop for', m[1]
        print "\nStarting Walk Forward run on", metaData['data_type'], "data..."
        if feature_selection == 'Univariate':
            print "Using top %i %s features" % (nfeatures, feature_selection)
        else:
            print "Using %s features" % feature_selection
        if longMemory == False:
            print "%i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        else:
            print "long memory starting with %i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        #cm_y_train = np.array([])
        cm_y_test_ga = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos_ga = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        leftoverIndex = nrows_oos%wfStep
        
        #reverse index to equate the wf tests of different periods, count backwards from the end
        wfIndex = range(nrows_oos-wfStep,-wfStep,-wfStep)
        tt_index =[]
        for i in wfIndex:
            #last wf index adjust the test index, else step
            if leftoverIndex > 0 and i == wfIndex[-1]:
                train_index = range(0,wf_is_period)        
                test_index = range(wf_is_period,wf_is_period+leftoverIndex)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
            else:
                if longMemory == True:
                    train_index = range(0,wf_is_period+i)
                else:
                    train_index = range(i,wf_is_period+i)
                test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
        #c=0
        for train_index,test_index in tt_index:
            #c+=1
            X_train, X_test = dX[train_index], dX[test_index]
            y_train_ga, y_test_ga = dy[train_index], dy[test_index]
            #print mmData_v.dates.iloc[test_index[-1]],
            #print 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
            #        'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
            #print train_index, test_index
            #check if there are no intersections
            intersect = np.intersect1d(datay_signal.reset_index().iloc[test_index]['index'].values,\
                        datay_signal.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
            if feature_selection == 'RFECV':
                #Recursive feature elimination with cross-validation: 
                #A recursive feature elimination example with automatic tuning of the
                #number of features selected with cross-validation.
                rfe = RFECV(estimator=RFE_estimator, step=1)
                rfe.fit(X_train, y_train_ga)
                #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                print 'Top %i RFECV features' % len(featureRank)
                print featureRank    
                metaData['featureRank'] = str(featureRank)
                X_train = rfe.transform(X_train)
                X_test = rfe.transform(X_test)
            else:
                #Univariate feature selection
                skb = SelectKBest(f_regression, k=nfeatures)
                skb.fit(X_train, y_train_ga)
                #dX_all = np.vstack((X_train.values, X_test.values))
                #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                #dX_v_rfe = X_new[dX_t.shape[0]:]
                X_train = skb.transform(X_train)
                X_test = skb.transform(X_test)
                featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                metaData['featureRank'] = str(featureRank)
                #print 'Top %i univariate features' % len(featureRank)
                #print featureRank
    
            #  fit the model to the in-sample data
            m[1].fit(X_train, y_train_ga)
            #trained_models[m[0]] = pickle.dumps(m[1])
                        
            #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
            y_pred_oos_ga = m[1].predict(X_test)
    
            if m[0][:2] == 'GA':
                print featureRank
                print '\nProgram:', m[1]._program
                #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
            
            #cm_y_train = np.concatenate([cm_y_train,y_train])
            cm_y_test_ga = np.concatenate([cm_y_test_ga,y_test_ga])
            #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
            cm_y_pred_oos_ga = np.concatenate([cm_y_pred_oos_ga,y_pred_oos_ga])
            #cm_train_index = np.concatenate([cm_train_index,train_index])
            cm_test_index = np.concatenate([cm_test_index,test_index])
            

        #create signals 1 and -1
        cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
        cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
        
        #if data is filtered so need to fill in the holes. signal = 0 for days that filtered
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
                
        #compute car, show matrix if data is filtered
        if data_type != 'ALL':
            print 'Metrics for filtered Validation Datapoints'
            prior_index_filt = pd.concat([st_oos_filt,unfilteredData.prior_index], axis=1,\
                                join='inner').prior_index.values.astype(int)
            #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                    ticker, validationFirstYear, validationFinalYear, iterations, metaData['filter'],showPDFCDF)
            CAR25_oos = CAR25_df(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=PRT['horizon'], DD95_limit =PRT['DD95_limit'])
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
                                    
        #add column prior index and gA.  if there are holes, nan values in signals
        st_oos_filt = pd.concat([st_oos_filt,unfilteredData.gainAhead,unfilteredData.prior_index],\
                                    axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
        #fills nan with zeros
        st_oos_filt['signals'].fillna(0, inplace=True)
        
        #fill zeros with opposite of input signal, if there are zeros. to return full data
        cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,metaData['input_signal']*-1,\
                                                                    st_oos_filt['signals'].values)
        cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
        #datay_gainAhead and cmatrix_test_index have the same index
        datay_gainAhead = st_oos_filt.gainAhead
        cmatrix_test_index = st_oos_filt.reset_index().index

        #plot learning curve, knn insufficient neighbors
        if showLearningCurve:
            try:
                plot_learning_curve(m[1], m[0], X_train,y_train_ga, scoring='r2')        
            except:
                pass
            
        #plot out-of-sample data
        if showPDFCDF:
            plt.figure()
            coef, b = np.polyfit(cm_y_pred_oos_ga, cm_y_test_ga, 1)
            plt.title('Out-of-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(cm_y_pred_oos_ga, cm_y_test_ga, '.')
            plt.plot(cm_y_pred_oos_ga, coef*cm_y_pred_oos_ga + b, '-')
            plt.show()
        
        #compute car, show matrix for all data is unfiltered
        if data_type == 'ALL':
            print 'Metrics for All Validation Datapoints'
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cmatrix_test_index, m[1], ticker,\
                                validationFirstYear, validationFinalYear, iterations, signal,showPDFCDF)
            CAR25_oos = CAR25_df(signal,cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, minFcst=PRT['horizon'], DD95_limit =PRT['DD95_limit'])
            #CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'LONG', 1)
            #CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
             #                       close, 'SHORT', -1)
        #update model metrics
        #metaData['signal'] = 'LONG 1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                                cmatrix_test_index, m, metaData,CAR25_oos)
        #metaData['signal'] = 'SHORT -1'
        #model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
        #                       cmatrix_test_index, m, metaData,CAR25_Sn1_oos)
    return model_metrics, st_oos_filt

    
def wf_regress_validate(unfilteredData, dataSet, models, model_metrics, wf_is_period, \
                           metaData, showPDFCDF=True, showLearningCurve=False, longMemory=False):
    close = unfilteredData.reset_index().Close
    #fill in the prior index. need this for the car25 calc uses the close index
    unfilteredData['prior_index'] = pd.concat([dataSet.prior_index, unfilteredData.Close],axis=1,join='outer').prior_index.interpolate(method='linear').dropna()
    ticker = metaData['ticker']
    data_type = metaData['data_type']
    iterations = metaData['iters']
    testFinalYear= metaData['t_end']
    validationFirstYear=metaData['v_start']
    validationFinalYear=metaData['v_end']
    wfStep=metaData['wf_step']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    tox_adj_proportion = metaData['tox_adj']
    feature_selection = metaData['FS']
    RFE_estimator = metaData['rfe_model'][1]
    metaData['rfe_model'] = metaData['rfe_model'][0]
    if 'filter' in metaData:
        filterName = metaData['filter']
    else:
        filterName = 'OOS_V'
    
    #r2 score is calculated each step. Where step >1. need to store values here when step >1.
    #metaData['r2_train'] = 'N/A'
    #metaData['r2_test'] = 'N/A'
    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']

    #check
    nrows_is = dataSet.ix[:testFinalYear].dropna().shape[0]
    if wf_is_period > nrows_is:
        print 'Walkforward insample period of', wf_is_period, 'is greater than in-sample data of ', nrows_is, '!'
        print 'Adjusting to', nrows_is, 'rows..'
        wf_is_period = nrows_is
    
    mmData = dataSet.ix[:testFinalYear].dropna()[-wf_is_period:]
    mmData_adj = adjustDataProportion(mmData, tox_adj_proportion)  #drop last row for hold days =1
    mmData_v = pd.concat([mmData_adj,dataSet.ix[validationFirstYear:validationFinalYear].dropna()], axis=0).reset_index()
    
    nrows_is = mmData.shape[0]
    nrows_oos = mmData_v.shape[0]-nrows_is
        
    metaData['rows'] = nrows_is
    
    #nrows = mmData_adj.shape[0]
    datay_signal = mmData_v[['signal', 'prior_index']]
    datay_gainAhead = mmData_v.gainAhead
    
    dataX = mmData_v.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    feature_names = []
    print '\nTotal %i features: ' % cols
    for i,x in enumerate(dataX.columns):
        print i,x+',',
        feature_names = feature_names+[x]
    if nfeatures > cols:
        print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
        print 'Adjusting to', cols, 'features..'
        nfeatures = cols  
            
    #  Copy from pandas dataframe to numpy arrays
    dy = np.zeros_like(datay_gainAhead)
    dX = np.zeros_like(dataX)
    
    dy = datay_gainAhead.values
    dX = dataX.values
    for m in models:
        print '\n\nNew WF train/predict loop for', m[1]
        print "\nStarting Walk Forward run on", metaData['data_type'], "data..."
        if feature_selection == 'Univariate':
            print "Using top %i %s features" % (nfeatures, feature_selection)
        else:
            print "Using %s features" % feature_selection
        if longMemory == False:
            print "%i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        else:
            print "long memory starting with %i rows in sample, %i rows out of sample, forecasting %i day(s) ahead.." % (nrows_is, nrows_oos,wfStep)
        #cm_y_train = np.array([])
        cm_y_test_ga = np.array([],dtype=float)
        #cm_y_pred_is = np.array([])
        cm_y_pred_oos_ga = np.array([],dtype=float)        
        cm_train_index = np.array([],dtype=int)
        cm_test_index = np.array([],dtype=int)
        
        leftoverIndex = nrows_oos%wfStep
        #wfIndex = range(0,nrows_oos,wfStep)
        #for i in wfIndex:
        #    #last wf index adjust the test index, else step
        #    if leftoverIndex > 0 and i == wfIndex[-1]:
        #        train_index = range(i,wf_is_period+i)
        #        #print train_index
        #        test_index = range(wf_is_period+i,wf_is_period+i+leftoverIndex)
        #        #print test_index
        #        X_train, X_test = dX[train_index], dX[test_index]
        #        y_train_ga, y_test_ga = dy[train_index], dy[test_index]
        #    else:
        #        train_index = range(i,wf_is_period+i)
        #        #print train_index
        #        test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
        #        #print test_index
        #        X_train, X_test = dX[train_index], dX[test_index]
        #        y_train_ga, y_test_ga = dy[train_index], dy[test_index]
        
        #reverse index to equate the wf tests of different periods, count backwards from the end
        wfIndex = range(nrows_oos-wfStep,-wfStep,-wfStep)
        tt_index =[]
        for i in wfIndex:
            #last wf index adjust the test index, else step
            if leftoverIndex > 0 and i == wfIndex[-1]:
                train_index = range(0,wf_is_period)        
                test_index = range(wf_is_period,wf_is_period+leftoverIndex)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
            else:
                if longMemory == True:
                    train_index = range(0,wf_is_period+i)
                else:
                    train_index = range(i,wf_is_period+i)
                test_index = range(wf_is_period+i,wf_is_period+i+wfStep)
                tt_index.insert(0,[train_index,test_index])
                #print i, 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
                #    'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
                #print train_index, test_index
        #c=0
        for train_index,test_index in tt_index:
            #c+=1
            X_train, X_test = dX[train_index], dX[test_index]
            y_train_ga, y_test_ga = dy[train_index], dy[test_index]
            #print mmData_v.dates.iloc[test_index[-1]],
            #print 't_start', mmData_v.dates.iloc[train_index[0]], 't_end', mmData_v.dates.iloc[train_index[-1]],\
            #        'v_start',mmData_v.dates.iloc[test_index[0]],'v_end', mmData_v.dates.iloc[test_index[-1]]
            #print train_index, test_index
            #check if there are no intersections
            intersect = np.intersect1d(datay_signal.reset_index().iloc[test_index]['index'].values,\
                        datay_signal.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
            if feature_selection == 'RFECV':
                #Recursive feature elimination with cross-validation: 
                #A recursive feature elimination example with automatic tuning of the
                #number of features selected with cross-validation.
                rfe = RFECV(estimator=RFE_estimator, step=1)
                rfe.fit(X_train, y_train_ga)
                #featureRank = [ feature_names[i] for i in rfe.ranking_-1]
                featureRank = [ feature_names[i] for i,b in enumerate(rfe.support_) if b==True]
                print 'Top %i RFECV features' % len(featureRank)
                print featureRank    
                metaData['featureRank'] = str(featureRank)
                X_train = rfe.transform(X_train)
                X_test = rfe.transform(X_test)
            else:
                #Univariate feature selection
                skb = SelectKBest(f_regression, k=nfeatures)
                skb.fit(X_train, y_train_ga)
                #dX_all = np.vstack((X_train.values, X_test.values))
                #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
                #dX_v_rfe = X_new[dX_t.shape[0]:]
                X_train = skb.transform(X_train)
                X_test = skb.transform(X_test)
                featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
                metaData['featureRank'] = str(featureRank)
                #print 'Top %i univariate features' % len(featureRank)
                #print featureRank
    
            #  fit the model to the in-sample data
            m[1].fit(X_train, y_train_ga)
            #trained_models[m[0]] = pickle.dumps(m[1])
                        
            #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))              
            y_pred_oos_ga = m[1].predict(X_test)
    
            if m[0][:2] == 'GA':
                print featureRank
                print '\nProgram:', m[1]._program
                #print 'R^2:    ', m[1].score(X_test_all,y_test_all) 
            
            #cm_y_train = np.concatenate([cm_y_train,y_train])
            cm_y_test_ga = np.concatenate([cm_y_test_ga,y_test_ga])
            #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
            cm_y_pred_oos_ga = np.concatenate([cm_y_pred_oos_ga,y_pred_oos_ga])
            #cm_train_index = np.concatenate([cm_train_index,train_index])
            cm_test_index = np.concatenate([cm_test_index,test_index])
            

        #create signals 1 and -1
        cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
        cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
        
        #if data is filtered so need to fill in the holes. signal = 0 for days that filtered
        st_oos_filt= pd.DataFrame()
        st_oos_filt['signals'] =  pd.Series(cm_y_pred_oos)
        st_oos_filt.index = mmData_v['dates'].iloc[cm_test_index]
        

        
        #compute car, show matrix if data is filtered
        if data_type != 'ALL':
            print 'Metrics for filtered Validation Datapoints'
            prior_index_filt = pd.concat([st_oos_filt,unfilteredData.prior_index], axis=1,\
                                join='inner').prior_index.values.astype(int)
            #datay_gainAhead and cm_test_index have the same index. dont need to have same shape because iloc is used in display
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cm_test_index, m[1],\
                    ticker, validationFirstYear, validationFinalYear, iterations, metaData['filter'],showPDFCDF)
            CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'LONG', 1)
            CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, prior_index_filt, close, 'SHORT', -1)
                                    
        #add column prior index and gA.  if there are holes, nan values in signals
        st_oos_filt = pd.concat([st_oos_filt,unfilteredData.gainAhead,unfilteredData.prior_index],\
                                    axis=1, join='outer').ix[validationFirstYear:validationFinalYear]
        #fills nan with zeros
        st_oos_filt['signals'].fillna(0, inplace=True)
        
        #fill zeros with opposite of input signal, if there are zeros. to return full data
        cm_y_pred_oos = np.where(st_oos_filt['signals'].values==0,metaData['input_signal']*-1,\
                                                                    st_oos_filt['signals'].values)
        cm_y_test = np.where(st_oos_filt.gainAhead>0,1,-1)
        #datay_gainAhead and cmatrix_test_index have the same index
        datay_gainAhead = st_oos_filt.gainAhead
        cmatrix_test_index = st_oos_filt.reset_index().index

        #plot learning curve, knn insufficient neighbors
        if showLearningCurve:
            try:
                plot_learning_curve(m[1], m[0], X_train,y_train_ga, scoring='r2')        
            except:
                pass
            
        #plot out-of-sample data
        if showPDFCDF:
            plt.figure()
            coef, b = np.polyfit(cm_y_pred_oos_ga, cm_y_test_ga, 1)
            plt.title('Out-of-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(cm_y_pred_oos_ga, cm_y_test_ga, '.')
            plt.plot(cm_y_pred_oos_ga, coef*cm_y_pred_oos_ga + b, '-')
            plt.show()
        
        #compute car, show matrix for all data is unfiltered
        if data_type == 'ALL':
            print 'Metrics for All Validation Datapoints'
            oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead, cmatrix_test_index, m[1], ticker,\
                                validationFirstYear, validationFinalYear, iterations, signal,showPDFCDF)
            CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, 'LONG', 1)
            CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, st_oos_filt['prior_index'].values.astype(int),\
                                    close, 'SHORT', -1)
        #update model metrics
        metaData['signal'] = 'LONG 1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                                cmatrix_test_index, m, metaData,CAR25_L1_oos)
        metaData['signal'] = 'SHORT -1'
        model_metrics = update_report(model_metrics, filterName, cm_y_pred_oos, cm_y_test, datay_gainAhead,\
                                cmatrix_test_index, m, metaData,CAR25_Sn1_oos)
    return model_metrics, st_oos_filt

def sss_regress_train(unfilteredData, dataSet, models, model_metrics, metaData, plotGA=True, showPDFCDF=0):
    # signal 1: beLong, 0: beFlat, -1: beShort
    close = unfilteredData.reset_index().Close
    ticker = metaData['ticker']
    iterations = metaData['iters']
    test_split = metaData['test_split']
    tox_adj_proportion = metaData['tox_adj']
    testFirstYear = metaData['t_start']
    testFinalYear = metaData['t_end']
    signal =  metaData['signal']
    nfeatures = metaData['n_features']
    dropCol = ['Open','High','Low','Close', 'Volume','gainAhead','signal','dates', 'prior_index']
    
    mmData = dataSet.ix[testFirstYear:testFinalYear].dropna().reset_index()   
    
    # remove last index for CAR 25 calc
    datay_gainAhead_all = mmData[:-1].gainAhead
    mmData_adj = adjustDataProportion(mmData.iloc[:-1], tox_adj_proportion)  #drop last row for hold days =1
    nrows = mmData_adj.shape[0]
    metaData['rows']=nrows
    
    # needs prior index for CAR25 calc
    datay = mmData_adj[['signal', 'prior_index']]
    datay_gainAhead = mmData_adj.gainAhead

    dataX = mmData_adj.drop(dropCol, axis=1) 
    cols = dataX.columns.shape[0]
    metaData['cols']=cols
    feature_names = []
    print '\nTotal %i features: ' % cols
    for i,x in enumerate(dataX.columns):
        print i,x+',',
        feature_names = feature_names+[x]
    if nfeatures > cols:
        print 'nfeatures', nfeatures, 'is greater than total features ', cols, '!'
        print 'Adjusting to', cols, 'features..'
        nfeatures = cols
    
    #  Copy from pandas dataframe to numpy arrays
    dy_sig = np.zeros_like(datay.signal.values)
    dX = np.zeros_like(dataX)

    dy_sig = datay.signal.values
    dX = dataX.values
          
    #  Make 'iterations' index vectors for the train-test split
    sss = StratifiedShuffleSplit(dy_sig,iterations,test_size=test_split, random_state=None)
    
    for i,m in enumerate(models):
        print '\nTraining', m[0]
        print "\nTesting Training Set %i rows.." % nrows

        cm_y_train_ga = np.array([])
        cm_y_test_ga = np.array([])
        #cm_y_pred_is = np.array([])
        #cm_y_pred_oos = np.array([])
        cm_y_pred_is_ga = np.array([])
        cm_y_pred_oos_ga = np.array([])        
        cm_train_index = np.array([])
        cm_test_index = np.array([])
        
        print 'New SSS train/predict loop for\n', m[1]
        for train_index,test_index in sss:
            X_train, X_test = dX[train_index], dX[test_index]
            y_train_ga, y_test_ga = datay_gainAhead[train_index], datay_gainAhead[test_index]
            
            #Univariate feature selection
            skb = SelectKBest(f_regression, k=nfeatures)
            skb.fit(X_train, y_train_ga)
            #dX_all = np.vstack((X_train.values, X_test.values))
            #dX_t_rfe = X_new[range(0,dX_t.shape[0])]
            #dX_v_rfe = X_new[dX_t.shape[0]:]
            X_train = skb.transform(X_train)
            X_test = skb.transform(X_test)
            featureRank = [ feature_names[i] for i in skb.get_support(feature_names)]
            metaData['featureRank'] = str(featureRank)
            print 'Top %i univariate features' % len(featureRank)
            print featureRank 
            
            #compare train&test index values, check if there are no intersections 
            intersect = np.intersect1d(datay.reset_index().iloc[test_index]['index'].values,\
                        datay.reset_index().iloc[train_index]['index'].values)
            if intersect.size != 0:
                print "\nDuplicate indexes found in test/training set: Possible Future Leak!"
       
            #  fit the model to the in-sample data
            m[1].fit(X_train, y_train_ga)
            #trained_models[m[0]] = pickle.dumps(m[1])
            #metaData['r2_train'] = m[1].score(X_train,y_train_ga)
            
            if m[0][:2] == 'GA':
                print '\nProgram:', m[1]._program
            
            plot_learning_curve(m[1], m[0], skb.transform(dX),datay_gainAhead, scoring='r2')
            y_pred_is_ga = m[1].predict(X_train)
            #y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))
            
            #oos test on all the data that excludes those in X_train, hold period [:-1]
            X_test_all = skb.transform(mmData.iloc[:-1].drop(dataX.reset_index()\
                        .iloc[train_index]['index'].values, axis=0)\
                        .drop(dropCol,axis=1).values)
            y_test_all_ga = mmData.iloc[:-1].drop(datay.reset_index()\
                        .iloc[train_index]['index'].values, axis=0).gainAhead.values
            test_index_all = mmData.iloc[:-1].drop(datay.reset_index().iloc[train_index]['index'].values, axis=0).index.values
            #metaData['r2_test'] = m[1].score(X_test_all,y_test_all_ga)
            y_pred_oos_ga = m[1].predict(X_test_all)
            #y_pred_oos = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_test_all)]))
      
            cm_y_train_ga = np.concatenate([cm_y_train_ga,y_train_ga])
            cm_y_test_ga = np.concatenate([cm_y_test_ga,y_test_all_ga])
            #cm_y_pred_is = np.concatenate([cm_y_pred_is,y_pred_is])
            #cm_y_pred_oos = np.concatenate([cm_y_pred_oos,y_pred_oos])
            cm_y_pred_is_ga = np.concatenate([cm_y_pred_is_ga,y_pred_is_ga])
            cm_y_pred_oos_ga = np.concatenate([cm_y_pred_oos_ga,y_pred_oos_ga])
            cm_train_index = np.concatenate([cm_train_index,train_index])
            cm_test_index = np.concatenate([cm_test_index,test_index_all])
            
        #convert gainAhead to signals
        cm_y_pred_is = np.array([-1 if x<0 else 1 for x in cm_y_pred_is_ga])
        cm_y_pred_oos = np.array([-1 if x<0 else 1 for x in cm_y_pred_oos_ga])
        cm_y_test = np.array([-1 if x<0 else 1 for x in cm_y_test_ga])
        cm_y_train = np.array([-1 if x<0 else 1 for x in cm_y_train_ga])
        
        if plotGA:
            #plot in-sample gainAhead
            plt.figure()
            coef, b = np.polyfit(cm_y_pred_is_ga, cm_y_train_ga, 1)
            plt.title('In-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(cm_y_pred_is_ga, cm_y_train_ga, '.')
            plt.plot(cm_y_pred_is_ga, coef*cm_y_pred_is_ga + b, '-')
            plt.show()
            #plot out-of-sample data
            plt.figure()
            coef, b = np.polyfit(cm_y_pred_oos_ga, cm_y_test_ga, 1)
            plt.title('Out-of-Sample')
            plt.ylabel('gainAhead')
            plt.xlabel('ypred gainAhead')
            plt.plot(cm_y_pred_oos_ga, cm_y_test_ga, '.')
            plt.plot(cm_y_pred_oos_ga, coef*cm_y_pred_oos_ga + b, '-')
            plt.show()
        
        is_display_cmatrix2(cm_y_train, cm_y_pred_is, datay_gainAhead, cm_train_index, m[1], ticker, testFirstYear, testFinalYear, iterations, signal,showPDFCDF)         
        CAR25_L1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[cm_train_index]['prior_index'].values.astype(int), close, 'LONG', 1)
        CAR25_Sn1_is = CAR25(signal, cm_y_pred_is, datay.reset_index().iloc[cm_train_index]['prior_index'].values.astype(int), close, 'SHORT', -1)
        model_metrics = update_report(model_metrics, "IS", cm_y_pred_is, cm_y_train, datay_gainAhead, cm_train_index, m, metaData,CAR25_L1_is)
        model_metrics = update_report(model_metrics, "IS", cm_y_pred_is, cm_y_train, datay_gainAhead, cm_train_index, m, metaData,CAR25_Sn1_is)
        
        oos_display_cmatrix2(cm_y_test, cm_y_pred_oos, datay_gainAhead_all, cm_test_index, m[1], ticker, testFirstYear, testFinalYear, iterations, signal,showPDFCDF)
        CAR25_L1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[cm_test_index]['prior_index'].values.astype(int), close, 'LONG', 1)
        CAR25_Sn1_oos = CAR25(signal, cm_y_pred_oos, datay.reset_index().iloc[cm_test_index]['prior_index'].values.astype(int), close, 'SHORT', -1)
        model_metrics = update_report(model_metrics, "OOS", cm_y_pred_oos, cm_y_test, datay_gainAhead_all, cm_test_index, m, metaData, CAR25_L1_oos)
        model_metrics = update_report(model_metrics, "OOS", cm_y_pred_oos, cm_y_test, datay_gainAhead_all, cm_test_index, m, metaData, CAR25_Sn1_oos)

    return model_metrics
    
def createBenchmark(sst,initialEquity,direction,start,end,ticker):
    #creates market benchmarks
    benchmarks = {}
    fract = pd.Series(data=1.0, index = sst.index, name='safef')
    
    #buyHold
    buyHoldSafef1 = pd.concat([pd.Series(data=1, index =  sst.index, name='signals'),\
                    sst.gainAhead, fract], axis=1)
    buyHoldSafef1.index = buyHoldSafef1.index.to_datetime()
    benchmarks[ticker+' buyHoldSafef1'] = calcEquity2(buyHoldSafef1.ix[start:end],initialEquity,'l')
    
    #sellHold
    sellHoldSafef1 = pd.concat([pd.Series(data=-1, index =  sst.index, name='signals'),\
                    sst.gainAhead, fract], axis=1)
    sellHoldSafef1.index = sellHoldSafef1.index.to_datetime()
    benchmarks[ticker+' sellHoldSafef1'] = calcEquity2(sellHoldSafef1.ix[start:end],initialEquity,'s')
    
    return benchmarks
    
def findBestDPS(DPS, PRT, system, start, end, direction, systemName, **kwargs):
    yscale=kwargs.get('yscale','log')
    ticker=kwargs.get('ticker',None)
    verbose=kwargs.get('verbose',True)
    displayCharts=kwargs.get('displayCharts',True)
    equityStatsSavePath=kwargs.get('equityStatsSavePath',None)
    v3tag=kwargs.get('v3tag',None)
    returnNoDPS=kwargs.get('returnNoDPS',True)
    savePath=kwargs.get('savePath',None)
    numCharts = kwargs.get('numCharts',None)
    equityCurves = {}
    DPS_adj = {}
    #f = [s for s in systems][0]
    if ticker==None:
        ticker = systemName[systemName.find('__')-4:systemName.find('__')]
    
    #set the start/end dates to ones that all equity curves share for apples to apples comparison.
    if type(start) == str:
        startDate = dt.strptime(start,'%Y-%m-%d')
    else:
        startDate = start
        
    if type(end) == str:
        endDate = dt.strptime(end,'%Y-%m-%d')
    else:
        endDate = end
        
    for dps in DPS:
        if DPS[dps].index[0]>startDate:
            startDate =  DPS[dps].index[0]
        if  DPS[dps].index[-1]<endDate:
            endDate =  DPS[dps].index[-1]
            
    DPS_adj = {}
    for dps in DPS:
        DPS_adj[dps] = DPS[dps].ix[startDate:]
        #round to nearest integer to reduce commissions
        DPS_adj[dps].safef = DPS_adj[dps].safef.round()
        
    for sst in DPS_adj:
        if verbose:
            print 'creating equity curve for ', sst
        zero_index = np.array([x for x in system.index if x not in DPS_adj[sst].index])
        sst_zero = pd.concat([pd.Series(data=0, name='signals', index=zero_index ), system.gainAhead.ix[zero_index],\
                pd.Series(data=0, name='safef', index=zero_index),pd.Series(data=0, name='CAR25', index=zero_index),\
                pd.Series(data=0, name='dd95', index=zero_index),pd.Series(data=0, name='ddTol', index=zero_index)],\
                axis=1).ix[startDate:]

        ec = calcEquity2(DPS_adj[sst], PRT['initial_equity'],sst[0])
        equityCurves[sst] = pd.concat([ec, sst_zero], axis=0).sort_index()
        #print DPS_adj[sst].tail(60)
        
        #DPS_adj[sst] = DPS_adj[sst].set_index(pd.DatetimeIndex(DPS_adj[sst]['index']))
        #DPS_adj[sst] = DPS_adj[sst].drop('index', axis=1)
        
    #create equity curves for safef1
    if returnNoDPS:
        if verbose:
            print 'creating equity curve for ', direction+systemName
        
        if direction == 'long':
            signals = pd.Series(data=np.where(system.signals == -1,0,1), name='signals',\
                        index= system.index).ix[startDate:]           
        elif direction == 'short':
            signals = pd.Series(data=np.where(system.signals == 1,0,-1), name='signals',\
                        index= system.index).ix[startDate:]
        else:
            signals = system.signals.ix[startDate:]

        system_sst = pd.concat([signals, system.gainAhead.ix[startDate:],\
                            pd.Series(data=system.prior_index.ix[startDate:], name = 'prior_index'),
                            pd.Series(data=1.0, name = 'safef', index = signals.index),
                            pd.Series(data=np.nan, name = 'CAR25', index = signals.index),
                            pd.Series(data=np.nan, name = 'dd95', index = signals.index),
                            pd.Series(data=np.nan, name = 'ddTol', index = signals.index),
                            ],axis=1)
        DPS_adj[direction+systemName] = system_sst
        equityCurves[direction+systemName] = calcEquity2(system_sst, PRT['initial_equity'],'b')
            
        
    #create equity curve stats    
    equityStats = pd.DataFrame(columns=['system','avgSafef','numTrades','cumCAR','MAXDD','sortinoRatio',\
                       'sharpeRatio','marRatio','k_ratio'], index = range(0,len(equityCurves)))
    #this calc dosent exclude non-trading days
    years_in_forecast = (endDate-startDate).total_seconds()/3600.0/365.0
    i=0
    for sst in equityCurves:
        avgSafef = equityCurves[sst].safef.mean()    
        numTrades = sum((equityCurves[sst].signals * equityCurves[sst].safef).round().diff().fillna(0).values !=0)
        cumCAR = 100*(((equityCurves[sst].equity.iloc[-1]/equityCurves[sst].equity.iloc[0])**(1.0/years_in_forecast))-1.0) 
        MAXDD = max(equityCurves[sst].maxDD)*-100.0
        sortinoRatio = ratio(equityCurves[sst].equity).sortino()
        sharpeRatio = ratio(equityCurves[sst].equity).sharpe()
        marRatio = cumCAR/-MAXDD
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(equityCurves[sst].equity.values)),equityCurves[sst].equity.values)
        k_ratio =(slope/std_err) * math.sqrt(252.0)/len(equityCurves[sst].equity.values)
        
        equityStats.iloc[i].system = sst
        equityStats.iloc[i].avgSafef = avgSafef
        equityStats.iloc[i].numTrades = numTrades
        equityStats.iloc[i].cumCAR = cumCAR
        equityStats.iloc[i].MAXDD = MAXDD
        equityStats.iloc[i].sortinoRatio = sortinoRatio
        equityStats.iloc[i].sharpeRatio = sharpeRatio
        equityStats.iloc[i].marRatio = marRatio
        equityStats.iloc[i].k_ratio = k_ratio
        i+=1

    #fill nan to zeros. happens when short system had no short signals
    equityStats = equityStats.fillna(0)
    #rank the curves based on scoring
    if direction == 'short':
        #short scoring weights drawdowns
        equityStats['avgSafefmm'] =minmax_scale(robust_scale(equityStats.avgSafef.reshape(-1, 1)))
        equityStats['numTradesmm'] =-minmax_scale(robust_scale(equityStats.numTrades.reshape(-1, 1)))
        equityStats['cumCARmm'] =minmax_scale(robust_scale(equityStats.cumCAR.reshape(-1, 1)))
        equityStats['MAXDDmm'] =minmax_scale(robust_scale(equityStats.MAXDD.reshape(-1, 1)))
        equityStats['sortinoRatiomm'] = minmax_scale(robust_scale(equityStats.sortinoRatio.reshape(-1, 1)))
        equityStats['marRatiomm'] =minmax_scale(robust_scale(equityStats.marRatio.reshape(-1, 1)))
        equityStats['sharpeRatiomm'] =minmax_scale(robust_scale(equityStats.sharpeRatio.reshape(-1, 1)))
        equityStats['k_ratiomm'] =minmax_scale(robust_scale(equityStats.k_ratio.reshape(-1, 1)))
        
        equityStats['scoremm'] =  equityStats.avgSafefmm+equityStats.cumCARmm+equityStats.MAXDDmm
        
        #equityStats['scoremm'] =  equityStats.avgSafefmm+equityStats.cumCARmm+equityStats.MAXDDmm+\
        #                                equityStats.sortinoRatiomm+equityStats.k_ratiomm+\
        #                               equityStats.sharpeRatiomm+equityStats.marRatiomm

        equityStats = equityStats.sort_values(['scoremm'], ascending=False)
    else:
        #long and both dps scoring. -#trades to reduce trading costs.
        equityStats['avgSafefmm'] =minmax_scale(robust_scale(equityStats.avgSafef.reshape(-1, 1)))
        equityStats['numTradesmm'] =-minmax_scale(robust_scale(equityStats.numTrades.reshape(-1, 1)))
        equityStats['cumCARmm'] =minmax_scale(robust_scale(equityStats.cumCAR.reshape(-1, 1)))
        #equityStats['MAXDDmm'] =minmax_scale(robust_scale(equityStats.MAXDD.reshape(-1, 1)))
        equityStats['sortinoRatiomm'] = minmax_scale(robust_scale(equityStats.sortinoRatio.reshape(-1, 1)))
        #equityStats['marRatiomm'] =minmax_scale(robust_scale(equityStats.marRatio.reshape(-1, 1)))
        #equityStats['sharpeRatiomm'] =minmax_scale(robust_scale(equityStats.sharpeRatio.reshape(-1, 1)))
        #equityStats['k_ratiomm'] =minmax_scale(robust_scale(equityStats.k_ratio.reshape(-1, 1)))

        equityStats['scoremm'] =  equityStats.avgSafefmm+equityStats.cumCARmm+\
                                        equityStats.sortinoRatiomm+equityStats.numTradesmm
                                       #equityStats.sharpeRatiomm
                                       #+equityStats.marRatiomm
                                       #+equityStats.MAXDDmm
                                        #equityStats.k_ratiomm+\
        equityStats = equityStats.sort_values(['scoremm'], ascending=False)
        
    topSystem = equityStats.system.iloc[0]

    if displayCharts or savePath is not None:
        if numCharts == None:
            numCharts = equityStats.system.shape[0]
            
        benchmarks = createBenchmark(equityCurves[topSystem],PRT['initial_equity'],'', startDate,endDate,ticker)
        benchStatsByYear = createYearlyStats(benchmarks)
        #create yearly stats for all equity curves with comparison against benchmark
        equityCurvesStatsByYear = createYearlyStats(equityCurves, benchStatsByYear)
        displayRankedCharts(numCharts,benchmarks,benchStatsByYear,equityCurves,equityStats,\
                                    equityCurvesStatsByYear, yscale=yscale, v3tag=v3tag,savePath=savePath, showPlot=displayCharts,\
                                    verbose=verbose)
                                    
    if equityStatsSavePath is not None:
        equityStats.to_csv(equityStatsSavePath+'eStats_'+systemName+'_'+str(endDate).replace(':','')+'.csv')
    return topSystem, DPS_adj[topSystem]
    
def createYearlyStats(eCurves, benchStatsByYear=None):
    yearlyStats = {}
    for df in eCurves:
        #print '\n'+df
        days = pd.Series(name='dataPoints')
        iEquity = pd.Series(name='iEquity')
        eEquity = pd.Series(name='eEquity')
        returns = pd.Series(name='return')
        acc = pd.Series(name='accuracy')
        numTrades = pd.Series(name='trades')
        for i, df_by_year in enumerate(eCurves[df].groupby(eCurves[df].index.year)):
            #dates
            year = df_by_year[0]
            days.set_value(year, df_by_year[1].shape[0])
            
            #returns
            iEquity.set_value(year, df_by_year[1].equity.iloc[0])
            eEquity.set_value(year, df_by_year[1].equity.iloc[-1])
            ret = (df_by_year[1].equity.iloc[-1]-df_by_year[1].equity.iloc[0])/df_by_year[1].equity.iloc[0]
            returns.set_value(year, ret)
            
            #accuracy
            ytrue = np.array([-1 if x<0 else 1 for x in df_by_year[1].gainAhead])
            ypred = df_by_year[1].signals.values
            acc_df = pd.concat([pd.Series(ytrue, name='ytrue'),pd.Series(ypred, name='ypred')],axis=1)
            ytrue = acc_df[acc_df.ypred != 0].ytrue.astype(int)
            ypred = acc_df[acc_df.ypred != 0].ypred.astype(int)
            if len(ytrue) == 0 or len(ypred)==0:
                acc.set_value(year, np.nan)
            else:
                acc.set_value(year, accuracy_score(ytrue,ypred))
            
            #trades
            nt = sum((df_by_year[1].signals * df_by_year[1].safef).round().diff().fillna(0).values !=0)
            numTrades.set_value(year, nt)
            
            #print year,df_by_year[1].shape, ret, accuracy_score(ytrue,ypred)
            
        yearlyStats[df] = pd.concat([days, numTrades, iEquity, eEquity, returns, acc], axis=1)
        
    if benchStatsByYear is None:
        return yearlyStats
    else:
        for df in yearlyStats:
            for bench_df in benchStatsByYear:
                return_diff = yearlyStats[df]['return']-benchStatsByYear[bench_df]['return']
                acc_diff = yearlyStats[df]['accuracy']-benchStatsByYear[bench_df]['accuracy']
                if bench_df[-13:] == 'buyHoldSafef1':
                    return_diff.name = 'retVs_buyHold'
                    acc_diff.name = 'accVs_buyHold'
                else: 
                    return_diff.name = 'retVs_topSystem'
                    acc_diff.name = 'accVs_topSystem'
                yearlyStats[df] =  pd.concat([yearlyStats[df], return_diff, acc_diff], axis =1)
        return yearlyStats
         
def calcEquity_df(SST, title, **kwargs):
    #leverage=1.0, equityCurveSavePath=None, pngPath=None, figsize=(8,7), showPlot=True
    leverage = kwargs.get('leverage',1.0)
    equityCurveSavePath = kwargs.get('equityCurveSavePath',None)
    pngPath = kwargs.get('pngPath',None)
    pngFilename = kwargs.get('pngFilename',None)
    figsize = kwargs.get('figsize',(8,7))
    showPlot =kwargs.get('showPlot',True)
    verbose = kwargs.get('verbose',True)
    initialEquity = kwargs.get('initialEquity',1.0)
    #version = kwargs.get('version',None)
    #v3tag = kwargs.get('v3tag',None)
    
    #initialEquity = 1.0
    nrows = SST.gainAhead.shape[0]
    #signalCounts = SST.signals.shape[0]
    '''
    if verbose:
        print '\nThere are %0.f signal counts' % nrows
        if 1 in SST.signals.value_counts():
            print SST.signals.value_counts()[1], 'beLong Signals',
        if -1 in SST.signals.value_counts():
            print SST.signals.value_counts()[-1], 'beShort Signals',
        if 0 in SST.signals.value_counts():
            print SST.signals.value_counts()[0], 'beFlat Signals',
    '''
        
    equityCurves = {}
    for trade in ['l','s','b']:       
        trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
        numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
        equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
        maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
        drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
        maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
        safef = pd.Series(data=leverage,index=range(0,len(SST.index)),name='safef')

        for i in range(0,len(SST.index)):
            if i == 0:
                equity[i] = initialEquity
                trades[i] = 0.0
                numBars[i] = 0.0
                maxEquity[i] = initialEquity
                drawdown[i] = 0.0
                maxDD[i] = 0.0

            else:
                if trade=='l':
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1 
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])

                        #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                elif trade=='s':
                    if (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    if (SST.signals[i-1] > 0):
                        trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    elif (SST.signals[i-1] < 0):
                        trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                        numBars[i] = numBars[i-1] + 1                
                        equity[i] = equity[i-1] + trades[i]
                        maxEquity[i] = max(equity[i],maxEquity[i-1])
                        drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                    else:
                        trades[i] = 0.0
                        numBars[i] = numBars[i-1]
                        equity[i] = equity[i-1]
                        maxEquity[i] = maxEquity[i-1]
                        drawdown[i] = drawdown[i-1]
                        maxDD[i] = max(drawdown[i],maxDD[i-1])
                        
        SSTcopy = SST.copy(deep=True)
        if trade =='l':
            #changeIndex = SSTcopy.signals[SST.signals==-1].index
            SSTcopy.loc[SST.signals==-1,'signals']=0
        elif trade =='s':
            #changeIndex = SSTcopy.signals[SST.signals==1].index
            SSTcopy.loc[SST.signals==1,'signals']=0
            
        equityCurves[trade] = pd.concat([SSTcopy.reset_index(), safef, trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)

    #  Compute cumulative equity for all days (buy and hold)   
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    safef = pd.Series(data=1.0,index=range(0,len(SST.index)),name='safef')
    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            numBars[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0
        else:
            trades[i] = safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
            numBars[i] = numBars[i-1] + 1 
            equity[i] = equity[i-1] + trades[i]
            maxEquity[i] = max(equity[i],maxEquity[i-1])
            drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
            maxDD[i] = max(drawdown[i],maxDD[i-1])          
    SSTcopy.loc[SST.signals==-1,'signals']=1
    SSTcopy.loc[SST.signals==0,'signals']=1
    equityCurves['buyHold'] = pd.concat([SSTcopy.reset_index(), safef, trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)
    
    #  Compute cumulative equity for all days (sell and hold)   
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    safef = pd.Series(data=1.0,index=range(0,len(SST.index)),name='safef')
    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            numBars[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0
        else:
            trades[i] = safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
            numBars[i] = numBars[i-1] + 1 
            equity[i] = equity[i-1] + trades[i]
            maxEquity[i] = max(equity[i],maxEquity[i-1])
            drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
            maxDD[i] = max(drawdown[i],maxDD[i-1])          
    SSTcopy.loc[SST.signals==1,'signals']=-1
    SSTcopy.loc[SST.signals==0,'signals']=-1
    equityCurves['sellHold'] = pd.concat([SSTcopy.reset_index(), safef, trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)
    
    if not SST.index.to_datetime()[0].time() and not SST.index.to_datetime()[1].time():
        barSize = '1 day'
    else:
        barSize = '1 min'
        
    #plt.close('all')
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=figsize)
    #plt.subplot(2,1,1)
    ind = np.arange(SST.shape[0])
    ax1.plot(ind, equityCurves['l'].equity,label="Long 1 Signals",color='b')
    ax1.plot(ind, equityCurves['s'].equity,label="Short -1 Signals",color='r')
    ax1.plot(ind, equityCurves['b'].equity,label="Long & Short",color='g')
    ax1.plot(ind, equityCurves['buyHold'].equity,label="BuyHold",ls='--',color='c')
    ax1.plot(ind, equityCurves['sellHold'].equity,label="SellHold",ls='--',color='lightpink')
    #fig, ax = plt.subplots(2)
    #plt.subplot(2,1,2)
    ax2.plot(ind, -equityCurves['l'].drawdown,label="Long 1 Signals",color='b')
    ax2.plot(ind, -equityCurves['s'].drawdown,label="Short -1 Signals",color='r')
    ax2.plot(ind, -equityCurves['b'].drawdown,label="Long & Short",color='g')
    ax2.plot(ind, -equityCurves['buyHold'].drawdown,label="BuyHold",ls='--',color='c')
    ax2.plot(ind, -equityCurves['sellHold'].drawdown,label="SellHold",ls='--',color='lightpink')
    
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)

    if barSize != '1 day' :
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d %H:%M")
        #ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    else:
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, SST.shape[0] - 1)
            return SST.index[thisind].strftime("%Y-%m-%d")
        #ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    
    
    minorLocator = MultipleLocator(SST.shape[0])
    ax1.xaxis.set_minor_locator(minorLocator)
    ax2.xaxis.set_minor_locator(minorLocator)
    ax1.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    ax1.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
    ax2.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
    # use a more precise date string for the x axis locations in the
    # toolbar

    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.set_title(title)
    ax1.set_ylabel("TWR")
    ax1.legend(loc="best")
    ax2.set_ylabel("Drawdown")
    #gets rid of space at the end
    ax1.set_xlim(0, SST.shape[0])
    ax2.set_xlim(0, SST.shape[0])
    #shows last date index
    xticks = ax1.xaxis.get_minor_ticks()
    xticks[1].label1.set_visible(False)
    xticks = ax2.xaxis.get_minor_ticks()
    xticks[1].label1.set_visible(False)
    
    #add text to chart
    text=  '\n%0.f bar counts: ' % nrows
    if 1 in SST.signals.value_counts():
        text+= '%i beLong,  ' % SST.signals.value_counts()[1]
    if -1 in SST.signals.value_counts():
        text+= '%i beShort,  ' % SST.signals.value_counts()[-1]
    if 0 in SST.signals.value_counts():
        text+= '%i beFlat  ' % SST.signals.value_counts()[0]
    shortTrades, longTrades = numberZeros(equityCurves['b'].signals)
    allTrades = sum((equityCurves['b'].signals * equityCurves['b'].safef).round().diff().fillna(0).values !=0)
    hoursTraded = (SST.index[-1]-SST.index[0]).total_seconds()/60.0/60.0
    avgTrades = float(allTrades)/hoursTraded
    #text+='\nValidation Period from %s to %s' % (str(SST.index[0]), str(SST.index[-1]))
    text+='\nAverage trades per hour: %0.2f' % (avgTrades)
    text+=  '\nTWR for Buy & Hold is %0.4f, %i Bars, maxDD %0.4f' %\
                (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
    text+=  '\nTWR for Sell & Hold is %0.4f, %i Bars, maxDD %0.4f' %\
                (equityCurves['sellHold'].equity.iloc[-1], nrows, equityCurves['sellHold'].maxDD.iloc[-1])
    text+='\nTWR for %i beLong trades is %0.4f, maxDD %0.4f' %\
                (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
    text+='\nTWR for %i beShort trades is %0.4f, maxDD %0.4f' %\
                (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
    text+='\nTWR for %i beLong and beShort trades with DPS is %0.4f, maxDD %0.4f' %\
                (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
    text+='\nAverage SAFEf: %0.4f' % (equityCurves['b'].safef.mean())
    plt.figtext(0.05,-0.15,text, fontsize=15)
    
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    if pngPath != None:
        if pngFilename==None:
            print 'Saving '+pngPath+title+'.png'
            plt.savefig(pngPath+title+'.png', bbox_inches='tight')
        else:
            print 'Saving '+pngPath+pngFilename+'.png'
            plt.savefig(pngPath+pngFilename+'.png', bbox_inches='tight')
    
    if showPlot:
        plt.show()
    plt.close(fig)
    
    shortTrades, longTrades = numberZeros(SST.signals)
    allTrades = sum((equityCurves['b'].signals * equityCurves['b'].safef).round().diff().fillna(0).values !=0)
    '''
    if verbose:
        hoursTraded = (SST.index[-1]-SST.index[0]).total_seconds()/60.0/60.0
        avgTrades = float(allTrades)/hoursTraded
        print '\nValidation Period from', SST.index[0],'to',SST.index[-1]
        print 'Average trades per hour: %0.2f' % (avgTrades)
        print 'TWR for Buy & Hold is %0.3f, %i Bars, maxDD %0.3f' %\
                    (equityCurves['buyHold'].equity.iloc[-1], nrows, equityCurves['buyHold'].maxDD.iloc[-1])
        print 'TWR for Sell & Hold is %0.3f, %i Bars, maxDD %0.3f' %\
                    (equityCurves['sellHold'].equity.iloc[-1], nrows, equityCurves['sellHold'].maxDD.iloc[-1])
        print 'TWR for %i beLong trades is %0.3f, maxDD %0.3f' %\
                    (longTrades, equityCurves['l'].equity.iloc[-1], equityCurves['l'].maxDD.iloc[-1])
        print 'TWR for %i beShort trades is %0.3f, maxDD %0.3f' %\
                    (shortTrades,equityCurves['s'].equity.iloc[-1], equityCurves['s'].maxDD.iloc[-1])
        print 'TWR for %i beLong and beShort trades with DPS is %0.3f, maxDD %0.3f' %\
                    (allTrades,equityCurves['b'].equity.iloc[-1], equityCurves['b'].maxDD.iloc[-1])
        print 'SAFEf:', equityCurves['b'].safef.mean()
    '''
    SST_equity = equityCurves['b']
    if 'dates' in SST_equity:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['dates'])).drop(['dates'], axis=1)
    else:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['index'])).drop(['index'], axis=1)

        
def calcEquity2(SST, initialEquity, trade):
    trades = pd.Series(data=0.0, index=range(0,len(SST.index)), name='trade')
    numBars = pd.Series(data=0.0, index=range(0,len(SST.index)), name='numBars')
    equity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='equity')
    maxEquity = pd.Series(data=0.0,index=range(0,len(SST.index)), name='maxEquity')
    drawdown = pd.Series(data=0.0,index=range(0,len(SST.index)), name='drawdown')
    maxDD = pd.Series(data=0.0,index=range(0,len(SST.index)),name='maxDD')
    
    for i in range(0,len(SST.index)):
        if i == 0:
            equity[i] = initialEquity
            trades[i] = 0.0
            numBars[i] = 0.0
            maxEquity[i] = initialEquity
            drawdown[i] = 0.0
            maxDD[i] = 0.0

        else:
            if trade=='l':
                if (SST.signals[i-1]*SST.safef[i-1] > 0):
                    trades[i] = SST.safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                    numBars[i] = numBars[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])

                    #print i, SST.signals[i], trades[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
                else:
                    trades[i] = 0.0
                    numBars[i] = numBars[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
            elif trade=='s':
                if (SST.signals[i-1]*SST.safef[i-1] < 0):
                    trades[i] = SST.safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                    numBars[i] = numBars[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    trades[i] = 0.0
                    numBars[i] = numBars[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
            else:
                if (SST.signals[i-1]*SST.safef[i-1] > 0):
                    trades[i] = SST.safef[i-1] * equity[i-1] * SST.gainAhead[i-1]
                    numBars[i] = numBars[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                elif (SST.signals[i-1]*SST.safef[i-1] < 0):
                    trades[i] = SST.safef[i-1] * equity[i-1] * -SST.gainAhead[i-1]
                    numBars[i] = numBars[i-1] + 1                
                    equity[i] = equity[i-1] + trades[i]
                    maxEquity[i] = max(equity[i],maxEquity[i-1])
                    drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                else:
                    trades[i] = 0.0
                    numBars[i] = numBars[i-1]
                    equity[i] = equity[i-1]
                    maxEquity[i] = maxEquity[i-1]
                    drawdown[i] = drawdown[i-1]
                    maxDD[i] = max(drawdown[i],maxDD[i-1])
                    
    SST_equity = pd.concat([SST.reset_index(), trades, numBars, equity,maxEquity,drawdown,maxDD], axis =1)
    
    if 'dates' in SST_equity:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['dates'])).drop(['dates'], axis=1)
    else:
        return SST_equity.set_index(pd.DatetimeIndex(SST_equity['index'])).drop(['index'], axis=1)
        
def calcEquity(sst, initialEquity):
    trade = pd.Series(data=0, index=range(0,len(sst.index)), name='trade')
    num_trades = pd.Series(data=0, index=range(0,len(sst.index)), name='numTrades')
    equity = pd.Series(index=range(0,len(sst.index)), name='equity')
    maxEquity = pd.Series(index=range(0,len(sst.index)), name='maxEquity')
    drawdown = pd.Series(index=range(0,len(sst.index)), name='drawdown')
    maxDD = pd.Series(index=range(0,len(sst.index)),name='maxDD')
    
    for i in range(0,len(sst.index)):
        if i == 0:
            equity[i] = initialEquity
            trade[i] = 0
            num_trades[i] = 0
            maxEquity[i] = initialEquity
            drawdown[i] = 0
            maxDD[i] = 0

        else:
            if (sst.signals[i-1] > 0):
                trade[i] = sst.safef[i-1] * equity[i-1] * sst.gainAhead[i-1]
                num_trades[i] = num_trades[i-1] + 1                
                equity[i] = equity[i-1] + trade[i]
                maxEquity[i] = max(equity[i],maxEquity[i-1])
                drawdown[i] = (maxEquity[i]-equity[i]) / maxEquity[i]
                maxDD[i] = max(drawdown[i],maxDD[i-1])

                #print i, sst.signals[i], trade[i], equity[i], maxEquity[i], drawdown[i], maxDD[i]
            else:
                trade[i] = 0.0
                num_trades[i] = num_trades[i-1]
                equity[i] = equity[i-1]
                maxEquity[i] = maxEquity[i-1]
                drawdown[i] = drawdown[i-1]
                maxDD[i] = max(drawdown[i],maxDD[i-1])
    sst_equity = pd.concat([sst.reset_index(), trade, num_trades, equity,maxEquity,drawdown,maxDD], axis =1)       
    return sst_equity.set_index(pd.DatetimeIndex(sst_equity['index'])).drop(['index'], axis=1)

def calcDPS2(signal_type, sst, PRT, start, end, windowLength, **kwargs):
    #threshold=kwargs.get('threshold',-np.inf)
    trade=kwargs.get('trade','long')
    asset = kwargs.get('asset','FX')
    def IBcommission(tradeAmount, asset):
        commission = 2.0
        if asset == 'FX':
            return max(2.0, tradeAmount*2e-5)
        else:
            return commission
            
    #updated: -1 short, 0 flat, 1 long
    if windowLength <=1:
        print 'windowLength needs to be >1 adjusting to 1.5'
        windowLength = 1.5
        
    windowLength = float(windowLength)
    initialEquity = PRT['initial_equity']
    ddTolerance = PRT['DD95_limit']
    tailRiskPct = PRT['tailRiskPct']
    forecastHorizon = PRT['horizon']
    maxLeverage = PRT['maxLeverage']
    minSafef= PRT['minSafef']
    threshold=PRT['CAR25_threshold']
    
    nCurves = 50
    accuracy_tolerance = 0.005
    updateInterval = 1
    #normalize personal risk tolerance by windowLength for safef
    multiplier = ddTolerance/math.sqrt(forecastHorizon) #assuming dd increases with sqrt of time
    forecastHorizon = windowLength/2.3 #need a bit more than 2x trades of the fcst horizon
    ddTolerance = math.sqrt(forecastHorizon)* multiplier #adjusted dd tolerance for the forecast
    years_in_forecast = (end-start).total_seconds()/3600.0/24.0/365.0
    #years_in_forecast = forecastHorizon / 252.0
    dpsRun = trade + signal_type + ' DPS wl%.1f maxL%i dd95_%.3f thres_%.1f' % (windowLength,maxLeverage,ddTolerance,threshold)
            
    print '\n', dpsRun, 'from', start, 'to', end
    #  Work with the index rather than the date
    startDate = sst.ix[start:end].index[0]
    endDate = sst.ix[start:end].index[-1]
    iStart = sst.index.get_loc(startDate)
    iEnd = sst.index.get_loc(endDate)
    
    safef_ser = pd.Series(index =  range(iStart,iEnd+1), name='safef')
    CAR25_ser = pd.Series(index =  range(iStart,iEnd+1), name='CAR25')
    dd95_ser = pd.Series(index =  range(iStart,iEnd+1), name='dd95')
    ddTol_ser = pd.Series(data=ddTolerance, index =  range(iStart,iEnd+1), name='ddTol')
    
    for i in range(iStart, iEnd+1, updateInterval):
        #print '.',
        #print '\n', i,sst.index[i], sst.signals[i], sst.gainAhead[i]
    
    #  Initialize variables
        curves = np.zeros(nCurves)
        numberDraws = np.zeros(nCurves)
        TWR = np.zeros(nCurves)
        maxDD = np.zeros(nCurves)
        
        fraction = 1.00
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
                            
                    commission = IBcommission(fraction*equity, asset)
                    thisTrade = fraction * tradeJ * equity    
                    equity = equity + thisTrade -commission
                    maxEquity = max(equity,maxEquity)
                    drawdown = (maxEquity-equity)/maxEquity
                    maxDrawdown = max(drawdown,maxDrawdown)
                    #print "thistrade, equity, signalsj, sst.gainAhead[i-j], tradej, fraction"
                    #print thisTrade, equity, signalsJ, sst.gainAhead[i-j], tradeJ, fraction
                    #print "maxDD, ndraws, horizonsofar, fcsthor:",maxDrawdown, nd, horizonSoFar, forecastHorizon
                #print nc, "\n\nCURVE DONE equity, maxDD, ndraws:", equity, maxDrawdown, nd      
                TWR[nc] = equity
                maxDD[nc] = maxDrawdown
                numberDraws[nc] = nd
        
            #  Find the drawdown at the tailLimit-th percentile        
            dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
            #print 'maxdd', maxDD
            #print "  DD %i: %.3f " % (tailRiskPct, dd95)
            fraction = fraction * ddTolerance / dd95
            #print fraction,
            TWR25 = stats.scoreatpercentile(TWR,25)        
            CAR25 = 100*(((TWR25/initialEquity) ** 
                      (1.0/years_in_forecast))-1.0)
            #print 'twr, car25,dd95, ddtol', TWR25, CAR25,dd95, ddTolerance
            #print 'dd95-ddtol', abs(ddTolerance-dd95)
    
            if fraction > maxLeverage:
                fraction = maxLeverage
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "fraction > maxLeverage" 
                done = True
            elif (abs(ddTolerance - dd95) < accuracy_tolerance):
                #print 'dd95-ddtol', abs(ddTolerance-dd95), 'accuracy_tolerance', accuracy_tolerance
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "Close enough" 
                done = True
        #threshold check
        if CAR25 > threshold:
            #print fraction, CAR25, dd95
            safef_ser[i] = fraction
            CAR25_ser[i] = CAR25
            dd95_ser[i] = dd95
        else:
            #dont trade
            safef_ser[i] = minSafef
            CAR25_ser[i] = CAR25
            dd95_ser[i] = threshold
    
    print "Done!"
    sst_save = sst.reset_index()
    sst_save = pd.concat([sst_save.ix[iStart:iEnd], safef_ser, CAR25_ser, dd95_ser, ddTol_ser], axis=1)
    #print sst_save.tail()
    #DPS[dpsRun] = sst_save
    if 'index'in sst_save:
        return dpsRun, sst_save.set_index(pd.DatetimeIndex(sst_save['index'])).drop(['index'], axis=1)
    else:
        return dpsRun, sst_save.set_index(pd.DatetimeIndex(sst_save['dates'])).drop(['dates'], axis=1)
        
def calcDPS(signal_type, sst, PRT, start, end, windowLength):
    windowLength = float(windowLength)
    initialEquity = PRT['initial_equity']
    ddTolerance = PRT['DD95_limit']
    tailRiskPct = PRT['tailRiskPct']
    forecastHorizon = PRT['horizon']
    maxLeverage = PRT['maxLeverage']
    nCurves = 100
    accuracy_tolerance = 0.005
    updateInterval = 1
    #normalize personal risk tolerance by windowLength for safef
    multiplier = ddTolerance/math.sqrt(forecastHorizon)
    forecastHorizon = windowLength/2.3
    ddTolerance = math.sqrt(forecastHorizon)* multiplier
    years_in_forecast = forecastHorizon / 252.0
    dpsRun = signal_type + ' DPS wl%i dd95_%.4f fcst_%idays' % (windowLength,ddTolerance,forecastHorizon)
            
    print '\n', dpsRun, 'from', start, 'to', end
    #  Work with the index rather than the date
    startDate = sst.ix[start:end].index[0]
    endDate = sst.ix[start:end].index[-1]
    iStart = sst.index.get_loc(startDate)
    iEnd = sst.index.get_loc(endDate)
    
    safef_ser = pd.Series(index =  range(iStart,iEnd), name='safef')
    CAR25_ser = pd.Series(index =  range(iStart,iEnd), name='CAR25')
    dd95_ser = pd.Series(index =  range(iStart,iEnd), name='dd95')
    
    for i in range(iStart, iEnd+1, updateInterval):
        print i,
        #print '\n', i,sst.index[i], sst.signals[i], sst.gainAhead[i]
    
    #  Initialize variables
        curves = np.zeros(nCurves)
        numberDraws = np.zeros(nCurves)
        TWR = np.zeros(nCurves)
        maxDD = np.zeros(nCurves)
        
        fraction = 1.00
        dd95 = 2 * ddTolerance
        done = False
        #print  "  Fraction ",
        while not done:
            #  Generate nCurve equity curves
            #print  fraction,
        
            for nc in range(nCurves):
        #        print "working on curve ", nc
                equity = initialEquity
                maxEquity = equity
                drawdown = 0
                maxDrawdown = 0
                horizonSoFar = 0
                nd = 0
                while (horizonSoFar < forecastHorizon):
                #get signals from starting from yesterday to windowlength
                    j = float(random.randint(1,windowLength+1))
                    #print j
                    nd = nd + 1
                    weightJ = 1.00 - j/windowLength
                    #print weightJ
                    horizonSoFar = horizonSoFar + weightJ
                    signalsJ = sst.signals[i-int(j)]
            #        print i, j, i-j, sst.iloc[i-int(j)]
                    if signalsJ > 0:
                        tradeJ = sst.gainAhead[i-int(j)] * weightJ
                    else:
                        tradeJ = 0.0
                    thisTrade = fraction * tradeJ * equity    
                    equity = equity + thisTrade
                    maxEquity = max(equity,maxEquity)
                    drawdown = (maxEquity-equity)/maxEquity
                    maxDrawdown = max(drawdown,maxDrawdown)
                    #print "signalsj, gainaheadj, tradej, fraction", signalsJ, sst.gainAhead[i-j], tradeJ, fraction
                    #print "thistrade, equity, maxDD, ndraws, horizonsofar, fcsthor:",\
                    #   thisTrade, equity, maxDrawdown, nd, horizonSoFar, forecastHorizon
                #print "WHILE DONE equity, maxDD, ndraws:", equity, maxDrawdown, nd        
                TWR[nc] = equity
                maxDD[nc] = maxDrawdown
                numberDraws[nc] = nd
        
            #  Find the drawdown at the tailLimit-th percentile        
            dd95 = stats.scoreatpercentile(maxDD,tailRiskPct)
            #print 'maxdd', maxDD
            #print "  DD %i: %.3f " % (tailRiskPct, dd95)
            fraction = fraction * ddTolerance / dd95
            #print fraction,
            TWR25 = stats.scoreatpercentile(TWR,25)        
            CAR25 = 100*(((TWR25/initialEquity) ** 
                      (1.0/years_in_forecast))-1.0)
            #print 'twr, car25,dd95, ddtol', TWR25, CAR25,dd95, ddTolerance
            #print 'dd95-ddtol', abs(ddTolerance-dd95)
    
            if (abs(ddTolerance - dd95) < accuracy_tolerance):
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "Close enough" 
                done = True
            elif fraction == float('inf'):
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "No DD" 
                done = True
            elif dd95 == 1: #max loss
                #print '\n DD95: %.3f ddTol: %.3f ' % (dd95,ddTolerance), "Max Loss" 
                fraction = 0
                done == True 
        
        #print fraction, CAR25, dd95
        if fraction > maxLeverage:
            fraction = maxLeverage
        safef_ser[i] = fraction
        CAR25_ser[i] = CAR25
        dd95_ser[i] = dd95
    
    print "\nSaving..."
    sst_save = sst.reset_index()
    sst_save = pd.concat([sst_save.ix[iStart:iEnd], safef_ser, CAR25_ser, dd95_ser], axis=1)
    #print sst_save.tail()
    #DPS[dpsRun] = sst_save
    return dpsRun, sst_save.set_index(pd.DatetimeIndex(sst_save['index'])).drop(['index'], axis=1)

def sss_iterate_train(dX,dy,sss,m):
    #  Initialize the confusion matrix
    cm_sum_is = np.zeros((2,2))
    cm_sum_oos = np.zeros((2,2)) 
 

    #  For each entry in the set of splits, fit and predict
    for train_index,test_index in sss:
        X_train, X_test = dX[train_index], dX[test_index]
        y_train, y_test = dy[train_index], dy[test_index] 
    
        #  fit the model to the in-sample data
        m[1].fit(X_train, y_train)
        
        if m[0][:2] == 'GA':
            print '\nProgram:', m[1]._program
            print 'R^2:    ', m[1].score(X_test,y_test)
            #graph = pydot.graph_from_dot_data(m[1]._program.export_graphviz())
            #display_png(graph.create_png(), raw=True) 
    
        #  test the in-sample fit
        y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))
        cm_is = confusion_matrix(y_train, y_pred_is)
        cm_sum_is = cm_sum_is + cm_is

        #  test the out-of-sample data
        y_pred_oos = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_test)]))
        cm_oos = confusion_matrix(y_test, y_pred_oos)
        cm_sum_oos = cm_sum_oos + cm_oos
        
    return cm_sum_is, cm_sum_oos

def sss_iterate_train2(dX,dy,sss,m):
    #  Initialize the confusion matrix
    #cm_sum_is = np.zeros((2,2))
    #cm_sum_oos = np.zeros((2,2)) 
    cm_ypred_is = np.empty([0])
    cm_ytrue_is = np.empty([0])
    cm_y_pred_oos = np.empty([0])
    cm_ytrue_oos = np.empty([0])

    #  For each entry in the set of splits, fit and predict
    for train_index,test_index in sss:
        X_train, X_test = dX[train_index], dX[test_index]
        y_train, y_test = dy[train_index], dy[test_index] 
    
        #  fit the model to the in-sample data
        m[1].fit(X_train, y_train)
        
        if m[0][:2] == 'GA':
            print '\nProgram:', m[1]._program
            print 'R^2:    ', m[1].score(X_test,y_test)
            #graph = pydot.graph_from_dot_data(m[1]._program.export_graphviz())
            #display_png(graph.create_png(), raw=True) 
    
        #  test the in-sample fit
        #cm_is = confusion_matrix(y_train, y_pred_is)
        #cm_sum_is = cm_sum_is + cm_is
        y_pred_is = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_train)]))
        if (cm_ypred_is.shape[0] == 0 and cm_ytrue_is.shape[0] == 0):
            cm_ypred_is = y_pred_is
            cm_ytrue_is = y_train
        else:
            cm_ypred_is = np.vstack((cm_ypred_is,y_pred_is))
            cm_ytrue_is = np.vstack((cm_ytrue_is,y_train))
    
        #  test the out-of-sample data
        #cm_oos = confusion_matrix(y_test, y_pred_oos)
        #cm_sum_oos = cm_sum_oos + cm_oos
        y_pred_oos = np.array(([-1 if x<0 else 1 for x in m[1].predict(X_test)]))
        if (cm_ypred_oos.shape[0] == 0 and cm_ytrue_oos.shape[0] == 0):
            cm_ypred_oos = y_pred_oos
            cm_ytrue_oos = y_test
        else:
            cm_ypred_oos = np.vstack((cm_ypred_oos,y_pred_oos))
            cm_ytrue_oos = np.vstack((cm_ytrue_oos,y_test))
    
    return cm_ypred_is, cm_ytrue_is, cm_ypred_oos, cm_ytrue_oos
    
def SymbolicTransformerTest(gp, sss, dX,dy, models,ticker,testFirstYear, testFinalYear, iterations, signal ):                             
    for train_index,test_index in sss:
        X_train, X_test = dX[train_index], dX[test_index]
        y_train, y_test = dy[train_index], dy[test_index] 
        
        est = Ridge()
        est.fit(X_train, y_train)
        
        print '\nOriginal Feature R^2:',
        print est.score(X_train, y_train)    
        for i,m in enumerate(models):
            cm_sum_is, cm_sum_oos = sss_iterate_train(dX,dy,sss,m)
            sss_display_cmatrix(cm_sum_is, cm_sum_oos, m[0],ticker,testFirstYear, testFinalYear, iterations, signal)
    
        #new features         
        gp.fit(X_train, y_train)
        gp_features = gp.transform(dX)
        new_dX = np.hstack((dX, gp_features))
      
        est = Ridge()
        est.fit(gp_features[train_index], dy[train_index])    
        print '\nNew feature R^2: ',
        print est.score(gp_features[train_index], dy[train_index])
        for i,m in enumerate(models):
            cm_sum_is, cm_sum_oos = sss_iterate_train(gp_features,dy,sss,m)
            sss_display_cmatrix(cm_sum_is, cm_sum_oos, m[0],ticker,testFirstYear, testFinalYear, iterations, signal)
    
        #Combined  
        est = Ridge()
        est.fit(new_dX[train_index], dy[train_index])
        print '\nCombined feature R^2:',
        print est.score(new_dX[train_index], dy[train_index])
        for i,m in enumerate(models):
            cm_sum_is, cm_sum_oos = sss_iterate_train(new_dX,dy,sss,m)
            sss_display_cmatrix(cm_sum_is, cm_sum_oos, m[0],ticker,testFirstYear, testFinalYear, iterations, signal)

            
def WF_Validation(validationFirstYear, validationFinalYear, mData, models, trained_models, trained_models_with_pred,X_train,y_train, X_train_oos_with_pred, y_train_oos):
    #Validation
    print '\nBeginning validation period ', validationFirstYear, \
            'through ', validationFinalYear
    
    #  Select the date range to test
    
    
    mmData_v = mData.ix[validationFirstYear:validationFinalYear]
    
    datay_v = mmData_v.toxic
    mmData_v = mmData_v.drop(['toxic'],axis=1)
    dataX_v = mmData_v
    
    nrows = mmData_v.shape[0]
    
    print "Testing Validation Set %i rows.." % nrows
    
    #  Reset dX,Dy
    dy = np.zeros_like(datay_v)
    dX = np.zeros_like(dataX_v)
    
    dy = datay_v.values
    dX = dataX_v.values
    
    # create predictions using single models
    col = []
    for m in models:
        col.append(m[0])
    
    y_pred_oos_no_wf_no_pred = pd.DataFrame(columns=col)
    y_pred_oos_no_wf_with_pred = pd.DataFrame(columns=col)
    y_pred_oos_cons_wo_pred = pd.DataFrame(columns=col)
    y_pred_oos_cons_with_pred = pd.DataFrame(columns=col)
    
    for i in range(0,nrows):
        print i,
        if i == 0:
            for m in trained_models:
                model = pickle.loads(trained_models[m])
                y_pred_oos_wf = model.predict(dX[i])
                y_pred_oos_cons_wo_pred[m] = pd.Series(y_pred_oos_wf)
                y_pred_oos_no_wf_no_pred[m] = model.predict(dX)
                #store the last iteration of y's
    
            
            #create predictions with model predictions
            X_train_with_pred = np.concatenate((dX[i],y_pred_oos_cons_wo_pred.iloc[i].values), axis=0)
            X_train_with_pred_no_wf = np.concatenate((dX,y_pred_oos_no_wf_no_pred), axis=1)
            
            for m in trained_models_with_pred:
                model = pickle.loads(trained_models_with_pred[m])
                y_pred_oos_cons_with_pred[m] = pd.Series(y_pred_oos_wf)
    
                y_pred_oos_wf = model.predict(X_train_with_pred)
                y_pred_oos_no_wf_with_pred[m] = model.predict(X_train_with_pred_no_wf)
    
            
            X_train = np.vstack((X_train, dX[i]))
            y_train = np.append(y_train, dy[i])
                    
            X_train_oos_with_pred = np.vstack((X_train_oos_with_pred, X_train_with_pred))
            y_train_oos = np.append(y_train_oos, dy[i])
        else:
            y_pred_oos_wf = np.empty(len(models))
            for j,m in enumerate(models):      
                m[1].fit(X_train, y_train)
                y_pred_oos_wf[j] = m[1].predict(dX[i])
                #y_pred_oos_cons.drop([m[0]],axis=1)
                #y_pred_oos_cons_wo_pred[m[0]].set_value(i,y_pred_oos_wf[0])
                
            y_pred_oos_cons_wo_pred.loc[i] = y_pred_oos_wf  
            X_train_with_pred = np.concatenate((dX[i],y_pred_oos_cons_wo_pred.iloc[i].values), axis=0)
            
            y_pred_oos_wf = np.empty(len(models))
            for j,m in enumerate(models):      
                m[1].fit(X_train_oos_with_pred, y_train_oos)
                y_pred_oos_wf[j] = m[1].predict(X_train_with_pred)
                #y_pred_oos_cons_with_pred[m[0]].set_value(i,y_pred_oos_wf[0])
            y_pred_oos_cons_with_pred.loc[i] = y_pred_oos_wf
            
            X_train = np.vstack((X_train, dX[i]))
            y_train = np.append(y_train, dy[i])
            
            X_train_oos_with_pred = np.vstack((X_train_oos_with_pred, X_train_with_pred))
            y_train_oos = np.append(y_train_oos, dy[i])
            
    
    print '\n\nNo WF OOS without Pred..'
    for m in y_pred_oos_no_wf_no_pred:
        cm_oos = confusion_matrix(dy, y_pred_oos_no_wf_no_pred[m].values)
        oos_display_cmatrix(cm_oos.astype(float), m, ticker,validationFirstYear, validationFinalYear, iterations=1)
    
    print '\n\nNo WF OOS with Pred..'
    for m in y_pred_oos_no_wf_with_pred:
        cm_oos = confusion_matrix(dy, y_pred_oos_no_wf_with_pred[m].values)
        oos_display_cmatrix(cm_oos.astype(float), m, ticker,validationFirstYear, validationFinalYear, iterations=1)
    
    print '\n\nWF OOS without Pred..'
    for m in y_pred_oos_cons_wo_pred:
        cm_oos = confusion_matrix(dy, y_pred_oos_cons_wo_pred[m].values)
        oos_display_cmatrix(cm_oos.astype(float), m, ticker,validationFirstYear, validationFinalYear, iterations=1)
    
    print '\n\nWF OOS with Pred..'
    for m in y_pred_oos_cons_with_pred:
        cm_oos = confusion_matrix(dy, y_pred_oos_cons_with_pred[m].values)
        oos_display_cmatrix(cm_oos.astype(float), m, ticker,validationFirstYear, validationFinalYear, iterations=1)

def maxCAR25(*args):
    global_max = float('-inf')
    loc_max = float('-inf')
    if type(args) == list:
        args = args[0]
    for i,d in enumerate(args[0]):
        #print i,d
        for k, v in d.iteritems():
            if k == 'CAR25':
                loc_max = v
            if loc_max > global_max:
                outer_d = d
                key = k
                global_max = loc_max
    return outer_d


def CAR25_df(signal_type, signals, signal_index, Close, minFcst=1008, DD95_limit = .20):
    #edited to compute all trades.
    #forecast horizon minimum 1 year, or safef becomes very large
    hold_days = 1
    number_forecasts = 50
    fraction = 1.00
    accuracy_tolerance = 0.005
    #DD95_limit = .20
    initial_equity = 100000
    forecast_horizon = min(minFcst,signal_index.shape[0]) #4 years because start date is 2007
    years_in_forecast = forecast_horizon/252.0
        
    #percentOfYearInMarket = number_long_signals /(years_in_study*252.0)
    #number_signals = index.shape[0]
    number_trades = forecast_horizon / hold_days
    number_days = number_trades*hold_days
    account_balance = np.zeros(number_days+1, dtype=float) 
    max_IT_DD = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade equity
    FC_max_IT_DD = np.zeros(number_forecasts, dtype=float) # Max intra-trade drawdown
    FC_tr_eq = np.zeros(number_forecasts, dtype=float)     # Trade equity (TWR)
    FC_trades = np.zeros(number_forecasts, dtype=float)     
    FC_sortino = np.zeros(number_forecasts, dtype=float)     
    FC_sharpe = np.zeros(number_forecasts, dtype=float)
    
    # start loop

    done = False    
    while not done:
        done = True
        #print 'Using fraction: %.3f ' % fraction,
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
            #print "forecast ",i_forecast, " of ", number_forecasts
            
            #wait = raw_input("PRESS ENTER TO CONTINUE.")
            #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
            trades = 0
            lastSignal = 0
            #  for each trade
            for i_trade in range(0,number_trades):
                #print 'day', i_trade, 'of',number_trades
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                index = random.choice(range(0,len(signal_index)-1))
                TRADE = signals[index]
                
                if TRADE > 0:
                    direction = 'LONG'
                else:
                    direction = 'SHORT'
                    
                if TRADE: #<0 not toxic
                    entry_index = signal_index[index]
                    #  Process the trade, day by day
                    for i_day_in_trade in range(0,hold_days+1):
                        if i_day_in_trade==0: #day 0
                            #  Things that happen immediately 
                            #  after the close of the signal day
                            #  Initialize for the trade
                            entry_price = Close.ix[entry_index]
                            #print account_balance[i_day], fraction, buy_price
                            number_shares = account_balance[i_day] * \
                                            fraction / entry_price
                            share_dollars = number_shares * entry_price
                            cash = account_balance[i_day] - \
                                   share_dollars
                            #print 'BUY', buy_price, "buy price", number_shares, "num_shares", share_dollars, "share $", cash, "cash"
                        else: # day n
                            #print 'iday in trade', i_day_in_trade, 'iday', i_day, 'num trades', number_trades
                            #  Things that change during a 
                            #  day the trade is held
                            i_day = i_day + 1
                            j = entry_index + i_day_in_trade
                            #  Drawdown for the trade
                            if direction == 'LONG':
                                profit = number_shares * (Close.ix[j] - entry_price)
                            else:                            
                                profit = number_shares * (entry_price - Close.ix[j])
                            MTM_equity = cash + share_dollars + profit
                            IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                    / max_IT_Eq[i_day-1]
                            max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                    IT_DD)
                            max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                    MTM_equity)
                            account_balance[i_day] = MTM_equity
                            #print 'accountbal', account_balance[i_day],'profit', profit, 'itdd', IT_DD,'max_IT_DD[i_day]',max_IT_DD[i_day], 'max_ITEq', max_IT_Eq[i_day]
                        if i_day_in_trade==hold_days: # last day of forecast
                            #  Exit at the close
                            exit_price = Close.ix[j]
                            if lastSignal == 0 or TRADE != lastSignal:
                                trades += 1
                                #print trades, lastSignal, TRADE
                                lastSignal = TRADE
                            #  Check for end of forecast
                            if i_day >= number_days:
                                #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                                FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                                FC_tr_eq[i_forecast] = MTM_equity
                                FC_trades[i_forecast] = trades
                                #print 'maxitdd', max_IT_DD[i_day]
    
                else: # no trade
                    #print 'iday', i_day, 'num days', number_days, 'num trades', number_trades
                    MTM_equity = account_balance[i_day]
                    i_day = i_day + 1
                    IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                            / max_IT_Eq[i_day-1]
                    max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                            IT_DD)
                    max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                            MTM_equity)
                    account_balance[i_day] = MTM_equity
                    #print 'no_trade ', 'mtm', MTM_equity, 'itdd', IT_DD, 'max_IT_DD', max_IT_DD[i_day], 'max_IT_Eq', max_IT_Eq[i_day]
                    if i_day >= number_days:
                        #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                        FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                        FC_tr_eq[i_forecast] = MTM_equity
                        FC_trades[i_forecast] = trades
                        FC_sortino[i_forecast] = ratio(account_balance).sortino()
                        FC_sharpe[i_forecast] = ratio(account_balance).sharpe()
                        #print 'maxitdd', max_IT_DD[i_day]
      
        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile 
        #print 'maxdd ', FC_max_IT_DD       
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)    
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #print '  DD95: %.3f ' % DD_95, "Close enough" 
            done = True
        elif DD_95 == 0: #no drawdown
            fraction =  float('inf')
            done == True
        elif DD_95 == 1: #max loss
            fraction = 0
            done == True 
        else:
            #print '  DD95: %.3f ' % DD_95, "Adjust fraction from " , fraction,
            fraction = fraction * DD95_limit / DD_95
            #print 'to ', fraction
            done = False
    
    #  Report
    SIG = signal_type
    YIF =  forecast_horizon/252.0
    TPY = FC_trades.mean()/(forecast_horizon/252.0)
    print '\nSignal: ', SIG
    print 'Fcst Horizon (years): %.1f ' % YIF, 
    #print ' TotalTrades: %.0f ' % FC_trades.mean(), 
    print ' Avg. Signals/Yr: %.2f' % TPY
    
    IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
    IT_DD_100 = max(FC_max_IT_DD)
    print 'DD95:  %.3f ' % IT_DD_95,
    print 'DD100: %.3f ' %  IT_DD_100,
    
    SOR25 = stats.scoreatpercentile(FC_sortino,25)
    SHA25 = stats.scoreatpercentile(FC_sharpe,25)
    print 'SORTINO25: %.3f ' %  SOR25,
    print 'SHARPE25: %.3f ' % SHA25
    
    print 'SAFEf: %.3f ' % fraction,
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
    
    
    TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
    CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
    CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
    CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    
    print 'CAR25: %.2f ' % CAR_25,
    print 'CAR50: %.2f ' % CAR_50,
    print 'CAR75: %.2f ' % CAR_75
    metrics = {'C25sig':SIG, 'safef':fraction, 'CAR25':CAR_25, 'CAR50':CAR_50, 'CAR75':CAR_75,\
                'DD95':IT_DD_95, 'DD100':IT_DD_100, 'SOR25':SOR25, 'SHA25':SHA25, 'YIF':YIF, 'TPY':TPY}
    return metrics
    
def CAR25(signal_type, signals, signal_index, Close, direction, trade_signal):
    #forecast horizon minimum 1 year, or safef becomes very large
    hold_days = 1
    number_forecasts = 50
    fraction = 1.00
    accuracy_tolerance = 0.005
    DD95_limit = .20
    initial_equity = 100000
    forecast_horizon = min(252,signal_index.shape[0]) #this number should probably be tied to the DPS window length 
    years_in_forecast = forecast_horizon/252.0
    #percentOfYearInMarket = number_long_signals /(years_in_study*252.0)
    #number_signals = index.shape[0]
    number_trades = forecast_horizon / hold_days
    number_days = number_trades*hold_days
    account_balance = np.zeros(number_days+1, dtype=float) 
    max_IT_DD = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade equity
    FC_max_IT_DD = np.zeros(number_forecasts, dtype=float) # Max intra-trade drawdown
    FC_tr_eq = np.zeros(number_forecasts, dtype=float)     # Trade equity (TWR)
    FC_trades = np.zeros(number_forecasts, dtype=float)     
    FC_sortino = np.zeros(number_forecasts, dtype=float)     
    FC_sharpe = np.zeros(number_forecasts, dtype=float)
    
    # start loop

    done = False    
    while not done:
        done = True
        #print 'Using fraction: %.3f ' % fraction,
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
            #print "forecast ",i_forecast, " of ", number_forecasts
            
            #wait = raw_input("PRESS ENTER TO CONTINUE.")
            #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
            trades = 0
            #  for each trade
            for i_trade in range(0,number_trades):
                #print 'day', i_trade, 'of',number_trades
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                index = random.choice(range(0,len(signal_index)-1))
                if trade_signal == 1:
                    TRADE = signals[index] > 0
                else:
                    TRADE = signals[index] < 0 #trade if signal is -1, false otherwise
                if TRADE: #<0 not toxic
                    entry_index = signal_index[index]
                    #  Process the trade, day by day
                    for i_day_in_trade in range(0,hold_days+1):
                        if i_day_in_trade==0: #day 0
                            #  Things that happen immediately 
                            #  after the close of the signal day
                            #  Initialize for the trade
                            entry_price = Close.ix[entry_index]
                            #print account_balance[i_day], fraction, buy_price
                            number_shares = account_balance[i_day] * \
                                            fraction / entry_price
                            share_dollars = number_shares * entry_price
                            cash = account_balance[i_day] - \
                                   share_dollars
                            #print 'BUY', buy_price, "buy price", number_shares, "num_shares", share_dollars, "share $", cash, "cash"
                        else: # day n
                            #print 'iday in trade', i_day_in_trade, 'iday', i_day, 'num trades', number_trades
                            #  Things that change during a 
                            #  day the trade is held
                            i_day = i_day + 1
                            j = entry_index + i_day_in_trade
                            #  Drawdown for the trade
                            if direction == 'LONG':
                                profit = number_shares * (Close.ix[j] - entry_price)
                            else:                            
                                profit = number_shares * (entry_price - Close.ix[j])
                            MTM_equity = cash + share_dollars + profit
                            IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                    / max_IT_Eq[i_day-1]
                            max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                    IT_DD)
                            max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                    MTM_equity)
                            account_balance[i_day] = MTM_equity
                            #print 'accountbal', account_balance[i_day],'profit', profit, 'itdd', IT_DD,'max_IT_DD[i_day]',max_IT_DD[i_day], 'max_ITEq', max_IT_Eq[i_day]
                        if i_day_in_trade==hold_days: # last day of forecast
                            #  Exit at the close
                            exit_price = Close.ix[j]
                            trades += 1
                            #  Check for end of forecast
                            if i_day >= number_days:
                                #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                                FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                                FC_tr_eq[i_forecast] = MTM_equity
                                FC_trades[i_forecast] = trades
                                #print 'maxitdd', max_IT_DD[i_day]
    
                else: # no trade
                    #print 'iday', i_day, 'num days', number_days, 'num trades', number_trades
                    MTM_equity = account_balance[i_day]
                    i_day = i_day + 1
                    IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                            / max_IT_Eq[i_day-1]
                    max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                            IT_DD)
                    max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                            MTM_equity)
                    account_balance[i_day] = MTM_equity
                    #print 'no_trade ', 'mtm', MTM_equity, 'itdd', IT_DD, 'max_IT_DD', max_IT_DD[i_day], 'max_IT_Eq', max_IT_Eq[i_day]
                    if i_day >= number_days:
                        #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                        FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                        FC_tr_eq[i_forecast] = MTM_equity
                        FC_trades[i_forecast] = trades
                        FC_sortino[i_forecast] = ratio(account_balance).sortino()
                        FC_sharpe[i_forecast] = ratio(account_balance).sharpe()
                        #print 'maxitdd', max_IT_DD[i_day]
      
        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile 
        #print 'maxdd ', FC_max_IT_DD       
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)    
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #print '  DD95: %.3f ' % DD_95, "Close enough" 
            done = True
        elif DD_95 == 0: #no drawdown
            fraction =  float('inf')
            done == True
        elif DD_95 == 1: #max loss
            fraction = 0
            done == True 
        else:
            #print '  DD95: %.3f ' % DD_95, "Adjust fraction from " , fraction,
            fraction = fraction * DD95_limit / DD_95
            #print 'to ', fraction
            done = False
    
    #  Report
    SIG = direction+' '+str(trade_signal)
    YIF =  forecast_horizon/252.0
    TPY = FC_trades.mean()/(forecast_horizon/252.0)
    print '\nSignal: ', SIG,' ' +signal_type
    print 'Fcst Horizon (years): %.1f ' % YIF, 
    #print ' TotalTrades: %.0f ' % FC_trades.mean(), 
    print ' Avg. Signals/Yr: %.2f' % TPY
    
    IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
    IT_DD_100 = max(FC_max_IT_DD)
    print 'DD95:  %.3f ' % IT_DD_95,
    print 'DD100: %.3f ' %  IT_DD_100,
    
    SOR25 = stats.scoreatpercentile(FC_sortino,25)
    SHA25 = stats.scoreatpercentile(FC_sharpe,25)
    print 'SORTINO25: %.3f ' %  SOR25,
    print 'SHARPE25: %.3f ' % SHA25
    
    print 'SAFEf: %.3f ' % fraction,
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
    
    
    TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
    CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
    CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
    CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    
    print 'CAR25: %.2f ' % CAR_25,
    print 'CAR50: %.2f ' % CAR_50,
    print 'CAR75: %.2f ' % CAR_75
    metrics = {'C25sig':SIG, 'safef':fraction, 'CAR25':CAR_25, 'CAR50':CAR_50, 'CAR75':CAR_75,\
                'DD95':IT_DD_95, 'DD100':IT_DD_100, 'SOR25':SOR25, 'SHA25':SHA25, 'YIF':YIF, 'TPY':TPY}
    return metrics

def CAR25_prospector(forecast_horizon, accuracy, signal_type, signals, signal_index, Close, direction, trade_signal):
    #forecast horizon as an input
    hold_days = 1
    number_forecasts = 50
    fraction = 1.00
    accuracy_tolerance = 0.005
    DD95_limit = .20
    initial_equity = 100000
    #forecast_horizon = signal_index.shape[0]
    years_in_forecast = forecast_horizon/252.0
    #percentOfYearInMarket = number_long_signals /(years_in_study*252.0)
    #number_signals = index.shape[0]
    number_trades = forecast_horizon / hold_days
    number_days = number_trades*hold_days
    account_balance = np.zeros(number_days+1, dtype=float) 
    max_IT_DD = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade equity
    FC_max_IT_DD = np.zeros(number_forecasts, dtype=float) # Max intra-trade drawdown
    FC_tr_eq = np.zeros(number_forecasts, dtype=float)     # Trade equity (TWR)
    FC_trades = np.zeros(number_forecasts, dtype=float)     
    FC_sortino = np.zeros(number_forecasts, dtype=float)     
    FC_sharpe = np.zeros(number_forecasts, dtype=float)
    
    # start loop

    done = False    
    while not done:
        done = True
        #print 'Using fraction: %.3f ' % fraction,
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
            #print "forecast ",i_forecast, " of ", number_forecasts
            
            #wait = raw_input("PRESS ENTER TO CONTINUE.")
            #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
            trades = 0
            #  for each trade
            for i_trade in range(0,number_trades):
                #print 'day', i_trade, 'of',number_trades
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                index = random.choice(range(0,len(signal_index)-1))
                gainer_loser_random = np.random.random()  
                if gainer_loser_random < accuracy:
                    if trade_signal == 1:
                        TRADE = signals[index] > 0
                    else:
                        TRADE = signals[index] < 0
                else:
                    if trade_signal == 1:
                        TRADE = signals[index] < 0
                    else:
                        TRADE = signals[index] > 0
                if TRADE: #<0 not toxic
                    entry_index = signal_index[index]
                    #  Process the trade, day by day
                    for i_day_in_trade in range(0,hold_days+1):
                        if i_day_in_trade==0: #day 0
                            #  Things that happen immediately 
                            #  after the close of the signal day
                            #  Initialize for the trade
                            entry_price = Close[entry_index]
                            #print account_balance[i_day], fraction, buy_price
                            number_shares = account_balance[i_day] * \
                                            fraction / entry_price
                            share_dollars = number_shares * entry_price
                            cash = account_balance[i_day] - \
                                   share_dollars
                            #print 'BUY', buy_price, "buy price", number_shares, "num_shares", share_dollars, "share $", cash, "cash"
                        else: # day n
                            #print 'iday in trade', i_day_in_trade, 'iday', i_day, 'num trades', number_trades
                            #  Things that change during a 
                            #  day the trade is held
                            i_day = i_day + 1
                            j = entry_index + i_day_in_trade
                            #  Drawdown for the trade
                            if direction == 'LONG':
                                profit = number_shares * (Close[j] - entry_price)
                            else:                            
                                profit = number_shares * (entry_price - Close[j])
                            MTM_equity = cash + share_dollars + profit
                            IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                    / max_IT_Eq[i_day-1]
                            max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                    IT_DD)
                            max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                    MTM_equity)
                            account_balance[i_day] = MTM_equity
                            #print 'accountbal', account_balance[i_day],'profit', profit, 'itdd', IT_DD,'max_IT_DD[i_day]',max_IT_DD[i_day], 'max_ITEq', max_IT_Eq[i_day]
                        if i_day_in_trade==hold_days: # last day of forecast
                            #  Exit at the close
                            exit_price = Close[j]
                            trades += 1
                            #  Check for end of forecast
                            if i_day >= number_days:
                                #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                                FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                                FC_tr_eq[i_forecast] = MTM_equity
                                FC_trades[i_forecast] = trades
                                #print 'maxitdd', max_IT_DD[i_day]
    
                else: # no trade
                    #print 'iday', i_day, 'num days', number_days, 'num trades', number_trades
                    MTM_equity = account_balance[i_day]
                    i_day = i_day + 1
                    IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                            / max_IT_Eq[i_day-1]
                    max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                            IT_DD)
                    max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                            MTM_equity)
                    account_balance[i_day] = MTM_equity
                    #print 'no_trade ', 'mtm', MTM_equity, 'itdd', IT_DD, 'max_IT_DD', max_IT_DD[i_day], 'max_IT_Eq', max_IT_Eq[i_day]
                    if i_day >= number_days:
                        #print '##############ENDSAVE i_day', i_day, 'number_days', number_days
                        FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                        FC_tr_eq[i_forecast] = MTM_equity
                        FC_trades[i_forecast] = trades
                        FC_sortino[i_forecast] = ratio(account_balance).sortino()
                        FC_sharpe[i_forecast] = ratio(account_balance).sharpe()
                        #print 'maxitdd', max_IT_DD[i_day]
      
        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile 
        #print 'maxdd ', FC_max_IT_DD       
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)    
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #print '  DD95: %.3f ' % DD_95, "Close enough" 
            done = True
        elif DD_95 == 0: #no drawdown
            fraction =  float('inf')
            done == True
        elif DD_95 == 1: #max loss
            fraction = 0
            done == True 
        else:
            #print '  DD95: %.3f ' % DD_95, "Adjust fraction from " , fraction,
            fraction = fraction * DD95_limit / DD_95
            #print 'to ', fraction
            done = False
    
    #  Report
    SIG = direction+' '+str(trade_signal)
    YIF =  forecast_horizon/252.0
    TPY = FC_trades.mean()/(forecast_horizon/252.0)
    print 'Signal: ', SIG, ' ', signal_type
    print 'Fcst Horizon (years): %.1f ' % YIF, 
    #print ' TotalTrades: %.0f ' % FC_trades.mean(), 
    print ' Avg. Signals/Yr: %.2f' % TPY
    
    IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
    IT_DD_100 = max(FC_max_IT_DD)
    print 'DD95:  %.3f ' % IT_DD_95,
    print 'DD100: %.3f ' %  IT_DD_100,
    
    SOR25 = stats.scoreatpercentile(FC_sortino,25)
    SHA25 = stats.scoreatpercentile(FC_sharpe,25)
    print 'SORTINO25: %.3f ' %  SOR25,
    print 'SHARPE25: %.3f ' % SHA25
    
    print 'SAFEf: %.3f ' % fraction,
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
    
    
    TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
    CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
    CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
    CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    
    print 'CAR25: %.2f ' % CAR_25,
    print 'CAR50: %.2f ' % CAR_50,
    print 'CAR75: %.2f ' % CAR_75
    metrics = {'C25sig':SIG, 'Type':signal_type, 'safef':fraction, 'CAR25':CAR_25, 'CAR50':CAR_50, 'CAR75':CAR_75,\
                'DD95':IT_DD_95, 'DD100':IT_DD_100, 'SOR25':SOR25, 'SHA25':SHA25, 'YIF':YIF, 'TPY':TPY}
    return metrics
     
def CAR25_cv(index, signals, Close, DD95_limit, initial_equity, startYear, endYear):
    #  has triangular weighting, and window period
    hold_days = 1
    number_forecasts = 50
    fraction = 1.00
    accuracy_tolerance = 0.005
    forecast_horizon = index.shape[0]
    number_signals = index.shape[0]
    number_trades = forecast_horizon / hold_days
    number_days = number_trades*hold_days
    account_balance = np.zeros(number_days+1, dtype=float) 
    max_IT_DD = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade drawdown
    max_IT_Eq = np.zeros(number_days+1, dtype=float)     # Maximum Intra-Trade equity
    FC_max_IT_DD = np.zeros(number_forecasts, dtype=float) # Max intra-trade drawdown
    FC_tr_eq = np.zeros(number_forecasts, dtype=float)     # Trade equity (TWR)


    done = False    
    while not done:
        done = True
        #print 'Using fraction: %.3f ' % fraction,
        # -----------------------------
        #   Beginning a new forecast run
        for i_forecast in range(number_forecasts):
        #   Initialize for trade sequence
            i_day = 0    # i_day counts to end of forecast
            #  Daily arrays, so running history can be plotted
            # Starting account balance
            account_balance[0] = initial_equity
            # Maximum intra-trade equity
            max_IT_Eq[0] = account_balance[0]    
            max_IT_DD[0] = 0
        
            #  for each trade
            for i_trade in range(0,number_trades):
                #print i_trade, number_trades
                #  Select the trade and retrieve its index 
                #  into the price array
                #  gainer or loser?
                #  Uniform for win/loss
                signal_index = random.randrange(len(index))
                if signals[signal_index] > 0: # entry >0 long
                    entry_index = index[signal_index]
                else:
                    entry_index = -1
                #  pick a trade accordingly
                #  for long positions, test is â€œ<â€
                #  for short positions, test is â€œ>â€
                #if gainer_loser_random < system_accuracy:
                    #  choose a gaining trade
                    #beLong_index = np.random.random_integers(0,number_long_signals)
                    #entry_index = beLong[beLong_index]                
                    #gainer_index = np.random.random_integers(0,number_gainers)
                    #entry_index = gainer[gainer_index] 
                #else:
                    #  choose a losing trade
                    #beShort_index = np.random.random_integers(0,number_short_signals)
                    #entry_index = beShort[beShort_index]
                    #loser_index = np.random.random_integers(0,number_losers)
                    #entry_index = loser[loser_index]
                if entry_index != -1:
                    #  Process the trade, day by day
                    for i_day_in_trade in range(0,hold_days+1):
                        if i_day_in_trade==0:
                            #  Things that happen immediately 
                            #  after the close of the signal day
                            #  Initialize for the trade
                            buy_price = Close[entry_index]
                            number_shares = account_balance[i_day] * \
                                            fraction / buy_price
                            share_dollars = number_shares * buy_price
                            cash = account_balance[i_day] - \
                                   share_dollars
                        else:
                            #  Things that change during a 
        			 #  day the trade is held
                            i_day = i_day + 1
                            j = entry_index + i_day_in_trade
                            #  Drawdown for the trade
                            profit = number_shares * (Close[j] - buy_price)
                            MTM_equity = cash + share_dollars + profit
                            IT_DD = (max_IT_Eq[i_day-1] - MTM_equity) \
                                    / max_IT_Eq[i_day-1]
                            max_IT_DD[i_day] = max(max_IT_DD[i_day-1], \
                                    IT_DD)
                            max_IT_Eq[i_day] = max(max_IT_Eq[i_day-1], \
                                    MTM_equity)
                            account_balance[i_day] = MTM_equity
                        if i_day_in_trade==hold_days:
                            #  Exit at the close
                            sell_price = Close[j]
                            #  Check for end of forecast
                            if i_day >= number_days:
                                FC_max_IT_DD[i_forecast] = max_IT_DD[i_day]
                                FC_tr_eq[i_forecast] = MTM_equity
                
        #  All the forecasts have been run
        #  Find the drawdown at the 95th percentile        
        DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
        #print '  DD95: %.3f ' % DD_95
        if (abs(DD95_limit - DD_95) < accuracy_tolerance):
            #  Close enough 
            done = True
        else:
            #  Adjust fraction and make a new set of forecasts
            fraction = fraction * DD95_limit / DD_95
            done = False

    #  Report
    print 'SAFEf: %.3f ' % fraction,
    #IT_DD_25 = stats.scoreatpercentile(FC_max_IT_DD,25)        
    #IT_DD_50 = stats.scoreatpercentile(FC_max_IT_DD,50)        
    IT_DD_95 = stats.scoreatpercentile(FC_max_IT_DD,95)
    print 'DD95: %.3f ' % IT_DD_95,
    
    years_in_forecast = float(endYear) - float(startYear)
    #percentOfYearInMarket = number_long_signals /(years_in_study*252.0)
    TWR_25 = stats.scoreatpercentile(FC_tr_eq,25)        
    CAR_25 = 100*(((TWR_25/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_50 = stats.scoreatpercentile(FC_tr_eq,50)
    CAR_50 = 100*(((TWR_50/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    TWR_75 = stats.scoreatpercentile(FC_tr_eq,75)        
    CAR_75 = 100*(((TWR_75/initial_equity) ** (1.0/years_in_forecast))-1.0)#*percentOfYearInMarket
    
    print 'CAR25: %.2f ' % CAR_25,
    print 'CAR50: %.2f ' % CAR_50,
    print 'CAR75: %.2f ' % CAR_75
    
    return fraction, IT_DD_95, CAR_25, CAR_50, CAR_75
    
def adjustDataProportion2(mmData, proportion,verbose=False):
    #non-monte carlo methods to make training faster, giving higher weight to recent data
    if proportion != 0:
        nrows = mmData.shape[0]
        neg_count=len(mmData.loc[mmData['signal']==-1])
        pos_count=len(mmData.loc[mmData['signal']==1])
        mmData_adj = mmData.loc[mmData['signal']==1]
        nrows_adj = float(mmData_adj.shape[0])
        
        if (neg_count/pos_count) < proportion:
            if verbose:
                print '(neg_count/pos_count) < proportion, sampling with replacement..'
            #raise ValueError('(neg_count/pos_count) < proportion, cannot sample w/o replacement')
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample with replacement
                if mmData['signal'][add_index] ==-1:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
        else:
            if verbose:
                print '(neg_count/pos_count) > proportion, appending recent', pos_count, '-1 days..'
            #append recent neg_count -1 days
            lastNegDaysIndex = mmData.loc[mmData['signal']==-1].index[-pos_count:]
            mmData_adj = pd.concat([mmData_adj, mmData.iloc[lastNegDaysIndex]], axis=0).sort_index()
            '''
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample without
                if mmData['signal'][add_index] ==-1 and add_index not in mmData_adj.index:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
            '''
        if verbose:    
            print "Adjusted Training Set to %i rows..(proportion = %f) " % (mmData_adj.shape[0], proportion)   
        return mmData_adj
    else:
        return mmData
        
def adjustDataProportion(mmData, proportion,verbose=1):
    if proportion != 0:
        nrows = mmData.shape[0]
        neg_count=len(mmData.loc[mmData['signal']==-1])
        pos_count=len(mmData.loc[mmData['signal']==1])
        mmData_adj = mmData.loc[mmData['signal']==1]
        nrows_adj = float(mmData_adj.shape[0])
        
        if (neg_count/pos_count) < proportion:
            if verbose:
                print '(neg_count/pos_count) < proportion, sampling with replacement..'
            #raise ValueError('(neg_count/pos_count) < proportion, cannot sample w/o replacement')
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample with replacement
                if mmData['signal'][add_index] ==-1:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
        else:
            if verbose:
                print '(neg_count/pos_count) > proportion, sampling without replacement..'
            nt_count = 0
            x = True
            while x:
                add_index = random.choice(mmData.loc[mmData['signal']==-1].index) #sample without
                if mmData['signal'][add_index] ==-1 and add_index not in mmData_adj.index:
                    mmData_adj = mmData_adj.append(mmData.iloc[add_index])
                    nt_count = float(mmData_adj.loc[mmData_adj['signal']==-1]['signal'].count())
                    #print nt_count, nrows_adj, nt_count/nrows_adj
                    
                if (nt_count/nrows_adj) >= proportion:
                    x = False
        if verbose:    
            print "Adjusted Training Set to %i rows..(proportion = %f) " % (mmData_adj.shape[0], proportion)   
        return mmData_adj
    else:
        return mmData
