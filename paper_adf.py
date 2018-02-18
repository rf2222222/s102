import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:10:29 2016
3 mins - 2150 dp per request
10 mins - 630 datapoints per request
30 mins - 1025 datapoints per request
1 hour - 500 datapoint per request
@author: Hidemi
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFECV
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
import os
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
#import adfapi.s105 as astrat
import adfapi.s108_pt as astrat

import adfapi.s105 as s105
import adfapi.s102 as s102
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf
import seitoolz.bars as bars
import re
logging.basicConfig(filename='logs/paper_adf.log',level=logging.DEBUG)

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

debug=False

if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    debug=True

if debug:
    showDist =  True
    showPDFCDF = True
    showAllCharts = True
    perturbData = True
    scorePath = './debug/scored_metrics_'
    equityStatsSavePath = './debug/'
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'
else:
    showDist =  False
    showPDFCDF = False
    showAllCharts = False
    perturbData = False
    scorePath = None
    equityStatsSavePath = None
    signalPath = './data/signals/'
    dataPath = './data/from_IB/'

    
def gettrades(sysname):
    filename='./data/paper/ib_' + sysname + '_trades.csv'
    if os.path.isfile(filename):
        dataSet=pd.read_csv(filename)
        #sums up results to starting acct capital
        #dataSet['equitycurve'] = initialEquity + dataSet['realized_PnL'].cumsum()
        return dataSet

def refresh_paper(sysname):
    files=['./data/paper/c2_' + sysname + '_account.csv','./data/paper/c2_' + sysname + '_trades.csv', \
    './data/paper/ib_'+ sysname + '_portfolio.csv','./data/paper/c2_' + sysname + '_portfolio.csv',  \
    './data/paper/ib_' + sysname + '_account.csv','./data/paper/ib_' + sysname + '_trades.csv']
    for i in files:
        filename=i
        if os.path.isfile(filename):
            os.remove(filename)
            print ('Deleting ' + filename)

def get_results(sysname, pairs, files):
    
    ibdata=seigraph.generate_paper_ib_plot(sysname, 'Date', 20000)
    seigraph.generate_mult_plot(ibdata,['equitycurve','PurePLcurve'], 'Date', 'paper_' + sysname + 'ib', sysname + " IB ", 'Equity')
    
    data=seigraph.get_data(sysname, 'paper', 'ib', 'trades', 'times', 20000)
    seigraph.generate_mult_plot(data,['realized_PnL','PurePL'], 'times', 'paper_' + sysname + 'ib' + sysname+'PL', 'paper_' + sysname + 'ib' + sysname + ' PL', 'PL')
    
    data=seigraph.get_data_files(files, pairs, 'Close', 20000)
    seigraph.generate_plots(data, 'paper_' + sysname + 'Close', sysname + " Close Price", 'Close')
    

pairs=[]

sysname='ADF'
if len(sys.argv) > 1 and sys.argv[1] == 'BTC':
    
    pairs=[
           ['USD', 'BTC_USD', [1,'USD','IDEALPRO', 's105_BTC_USD']],
           ['KRW', 'BTC_KRW', [1,'USD','IDEALPRO', 's105_BTC_KRW']]]

elif len(sys.argv) > 1 and sys.argv[1] == 'JPY':
    
    pairs=[
           ['USD', 'BTC_USD', [1,'USD','IDEALPRO', 's107_BTC_USD']],
           ['JPY', 'BTCJPY', [1,'USD','IDEALPRO', 's107_BTCJPY']]]
    
    sysname='PAPER_ADF_XBT_JPY'

elif len(sys.argv) > 1 and sys.argv[1] == 'ETH':

    pairs=[
           ['USD', 'XBTUSD', [1,'USD','IDEALPRO', 's108_XBTUSD']],
           ['USD', 'ETHH18', [10,'USD','IDEALPRO', 's108_ETHH18']]]
    
    sysname='PAPER_ADF_XBT_ETH'

elif len(sys.argv) > 1 and sys.argv[1] == 'BTCBTC':

    pairs=[
           ['USD', 'XBTUSD', [1,'USD','IDEALPRO', 's108_XBTUSD']],
           ['USD', 'XBTH18', [1,'USD','IDEALPRO', 's108_XBTH18']]]
    
    sysname='PAPER_ADF_XBT_XBT'
        
elif len(sys.argv) > 1 and sys.argv[1] == 'XBTXBT':
    
    pairs=[
           ['USD', 'XBTH18', [1,'USD','IDEALPRO', 's108_XBTH18']],
           ['USD', 'XBTM18', [1,'USD','IDEALPRO', 's108_XBTM18']]]
    
    sysname='PAPER_ADF_XBT_XBT'
    
        
else:

    pairs=[
           ['USD', 'XBTH18', [1,'USD','IDEALPRO', 's108_XBTH18']],
           ['USD', 'XBTM18', [1,'USD','IDEALPRO', 's108_XBTM18']]]
    
    sysname='PAPER_ADF_XBT_XBT'


frequency=60
#data=gettrades(sysname)
SST=bars.get_bar_elastic_history(pairs, 'close', frequency)
print (SST)
if SST.shape[0] > 10000:
    SST=SST.tail(10000)
threads = []

pos=dict()
totalpos=dict()
asks=dict()
bids=dict()

def proc_pair(sysname, sym1, sym2, param1, param2):
       
        symPair=sym1+sym2
        
        refresh_paper(sysname)
        if not pos.has_key(symPair):
            pos[symPair]=dict()
        params=dict()
       
        params[sym1]=param1
        params[sym2]=param2
        confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
        print ("Coint Confidence: " + str(confidence) + "%")
        for i in SST.index:
            try:
                priceHist=SST.ix[i]
                asks[sym1]=priceHist[sym1]
                bids[sym1]=priceHist[sym1]
                asks[sym2]=priceHist[sym2]
                bids[sym2]=priceHist[sym2]
                timestamp=time.mktime(priceHist['timestamp'].timetuple())
                bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                if signals and len(signals) >= 1:
                    for signal in signals:
                        (barSym, barSig, barCmt)=signal
                        
                        if pos[symPair].has_key(barSym):
                            pos[symPair][barSym]=pos[symPair][barSym] + barSig
                        else:
                            pos[symPair][barSym]=barSig
                            
                        if totalpos.has_key(barSym):
                            totalpos[barSym]=totalpos[barSym] + barSig
                        else:
                            totalpos[barSym]=barSig
                        
                        model=generate_model_manual(barSym, totalpos[barSym], 1)
                        
                        if totalpos[barSym] == 0:
                            totalpos.pop(barSym, None)
                            
                        if pos[symPair][barSym] == 0:
                            pos[symPair].pop(barSym, None)
                            
                        (mult, currency, exchange, signalfile)=params[barSym]
                        commissionkey=barSym + currency + exchange
                        commission_pct=0.000625
                        commission_cash=0
                        if commissionkey in commissiondata.index:
                            commission=commissiondata.loc[commissionkey]
                            commission_pct=float(commission['Pct'])
                            commission_cash=float(commission['Cash'])
                            
                        ask=float(asks[barSym])
                        bid=float(bids[barSym])
                        
                        sym=barSym
                        ibsym=barSym
                        secType='BITCOIN'
                        ibsym=sym
                        
                        pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                        if ask > 0 and bid > 0:
                           
                            date=datetime.datetime.fromtimestamp(
                                int(timestamp)
                            ).strftime("%Y%m%d %H:%M:%S EST")
                            print ('Signal: ' + barSym + '[' + str(barSig) + ']@' + str(ask))
                            adj_size(model, barSym, sysname, pricefeed,   \
                                sysname,sysname,mult,barSym, secType, True, \
                                    mult, ibsym,currency,exchange, secType, True, date)
    
            except Exception as e:
                logging.error('proc_pair', exc_info=True)
        
seen=dict()
reslist=dict()
def proc_backtest(sysname, SST):
    for [file1,sym1, mult1] in pairs:
        #print "sym: " + sym1
        for [file2,sym2, mult2] in pairs:
            if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
                seen[sym1+sym2]=1
                seen[sym2+sym1]=1
                mypairs=[sym1,sym2]
                myfiles=[file1, file2]
                mysysname=sysname +'_'+ sym1+'_'+sym2
                reslist[mysysname]=[mypairs, myfiles]
                
                sig_thread = threading.Thread(target=proc_pair, args=[mysysname, sym1, sym2, mult1, mult2])
                sig_thread.daemon=True
                threads.append(sig_thread)
    [t.start() for t in threads]
    [t.join() for t in threads]

proc_backtest(sysname, SST)

for sysname in reslist.keys():
    (mypair,myfile)=reslist[sysname]
    get_results(sysname, mypair, myfile)


