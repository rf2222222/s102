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