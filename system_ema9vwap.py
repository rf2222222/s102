import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from bdb import bar
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
from paper.signal import get_dps_model_pos, get_model_pos, generate_model_manual, generate_model_sig, get_model_sig
from paper.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
#import adfapi.s105 as astrat
import adfapi.ema_vwap as astrat
#import adfapi.s102 as s102
#import paper.graph as seigraph
import adfapi.adf_helper as adf
from dateutil.parser import parse
import pytz


logging.basicConfig(filename='./system_ema9_vwap.log',level=logging.DEBUG)
path='/MQL4/Files/'
barpath='/MQL4/Files/bars/'
pairs=[]

pparams=dict()
pairs=[
       [path + '5m_EURUSD.csv', '5m_EURUSD', [100000,'USD','IDEALPRO', 'EMA9VWAP_EURUSD']],
       [path + '5m_EURUSD.csv', '5m_EURUSD', [100000,'USD','IDEALPRO', 'EMA9VWAP_EURUSD']],
       ]
if len(sys.argv) > 1 and sys.argv[1] == 'EURUSD':
    pairs=[
       [path + '5m_EURUSD.csv', '5m_EURUSD', [100000,'USD','IDEALPRO', 'EMA9VWAP_EURUSD']],
       [path + '5m_EURUSD.csv', '5m_EURUSD', [100000,'USD','IDEALPRO', 'EMA9VWAP_EURUSD']],
       ]
elif len(sys.argv) > 1 and sys.argv[1] == 'AAPL':
    pairs=[
       [path + '5m_#Apple.csv', '5m_#Apple', [100,'USD','IDEALPRO', 'EMA9VWAP_AAPL']],
       [path + '5m_#Apple.csv', '5m_#Apple', [100,'USD','IDEALPRO', 'EMA9VWAP_AAPL']],
       ]
elif len(sys.argv) > 1 and sys.argv[1] == 'BARC':
    pairs=[
       [path + '5m_#Barclays.csv', '5m_#Barclays', [100,'USD','IDEALPRO', 'EMA9VWAP_BARC']],
       [path + '5m_#Barclays.csv' , '5m_#Barclays', [100,'USD','IDEALPRO', 'EMA9VWAP_BARC']],
       ]


        
def get_last_bars(currencyPairs, ylabel, callback):
    global tickerId
    global lastDate
    while 1:
        try:
            
            SSTdata=pd.DataFrame()
            symbols=list()
            returnData=False
            for ticker in currencyPairs:
                pair=ticker
                minFile= path + '/' +pair+'.csv'
                symbol = pair
                date=''
                
                if os.path.isfile(minFile):
                    dta=pd.read_csv(minFile)
                    date=str(dta.iloc[-1]['Date'])
                    
                    eastern=timezone('US/Eastern')
                    date=parse(date).replace(tzinfo=eastern)
                    print date
                    timestamp = time.mktime(date.timetuple())
                    #print 'loading',minFile,date,dta[ylabel],'\n'
                    data=pd.DataFrame()
                    data['Date']=dta['Date']
                    data[symbol]=dta[ylabel]
                    data[str(symbol+ '_High')]=dta['High']
                    data[str(symbol+ '_Low')]=dta['Low']
                    data[str(symbol+ '_Volume')]=dta['Volume']
                    #data['timestamp']= data['Date']
                    data=data.set_index('Date') 
                    
                    if data.shape[0] > 0:
                        if SSTdata.shape[0] < 1:
                            SSTdata=data
                        else:
                            SSTdata = SSTdata.combine_first(data).sort_index()
                        
                        if not lastDate.has_key(symbol):
                            returnData=True
                            lastDate[symbol]=timestamp
                            symbols.append(symbol)
                                                   
                        #print lastDate[symbol], timestamp
                        if lastDate[symbol] < timestamp:
                            returnData=True
                            lastDate[symbol]=timestamp
                            symbols.append(symbol)
                        #print 'Shape: ' + str(len(SSTdata.index)) 
                        returnData=True
            
            if returnData:
                data=SSTdata.copy()
                
                data=data.reset_index() #.set_index('Date')
                data=data.fillna(method='pad')
                callback(data, symbols)
            time.sleep(20)
        except Exception as e:
            logging.error("get_last_bar", exc_info=True)
            
            

def get_bar_history(datas, ylabel):
    try:
        SST=pd.DataFrame()
        
        for (filename, ticker, qty) in datas:
            dta=pd.read_csv(filename)
            print 'Reading ',filename
            #symbol=ticker[0:3]
            #currency=ticker[3:6]
            #print 'plot for ticker: ' + currency
            #if ylabel == 'Close':
            #    diviser=dta.iloc[0][ylabel]
            #    dta[ylabel]=dta[ylabel] /diviser
                
            #dta[ylabel].plot(label=ticker)   
            data=pd.DataFrame()
            
            data['Date']=pd.to_datetime(dta[dta.columns[0]])
            
            data[ticker]=dta[ylabel]
            data[ticker+ '_High']=dta['High']
            data[ticker+ '_Low']=dta['Low']
            data[ticker+ '_Volume']=dta['Volume']
            data=data.set_index('Date') 
            if len(SST.index.values) < 2:
                SST=data
            else:
                SST = SST.combine_first(data).sort_index()
        colnames=list()
        for col in SST.columns:
            if col != 'Date' and col != 0:
                colnames.append(col)
        data=SST
        data=data.reset_index()        
        data['timestamp']= data['Date']
        
        data=data.set_index('Date')
        data=data.fillna(method='pad')
        return data
        
    except Exception as e:
        logging.error("something bad happened", exc_info=True)
    return SST



def prep_pair(sym1, sym2, param1, param2):
        global pos
        symPair=sym1+sym2
        if not pos.has_key(symPair):
            pos[symPair]=dict()

        params=dict()
        
        params[sym1]=param1
        params[sym2]=param2
        
        confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
        print "Coint Confidence: " + str(confidence) + "%"
        for i in SST.index:
            try:
                priceHist=SST.ix[i]
                
                timestamp=time.mktime(priceHist['timestamp'].timetuple())
                
                bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp), priceHist[sym1+'_High'], priceHist[sym1+'_Low'], priceHist[sym1+'_Volume'])
                bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp), priceHist[sym2+'_High'], priceHist[sym2+'_Low'], priceHist[sym2+'_Volume'])
                
                astrat.procBar(bar1, bar2, pos[symPair], False)
                #proc_signals(signals, params, symPair, timestamp)
                
                #signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                #proc_signals(signals, params, symPair, timestamp)
                
            except Exception as e:
                 logging.error('prep_pair', exc_info=True)
                

def proc_pair(sym1, sym2, param1, param2):
        #while 1:
        try:
            proc_onBar(sym1, sym2, param1, param2)
            #time.sleep(20)
        except Exception as e:
            logging.error("proc_pair", exc_info=True)

                
def proc_onBar(sym1, sym2, param1, param2):
        symPair=sym1+sym2
        
        params=dict()
        params[sym1]=param1
        params[sym2]=param2
    
        try:
            bardict[sym1]=get_bar(sym1)
            bardict[sym2]=get_bar(sym2)
            timestamp=int(time.time())
            date=datetime.datetime.fromtimestamp(
                timestamp
            ).strftime('%Y-%m-%d %H:%M:00') 
            bardate=time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:00").timetuple())
            
            if not lastDate.has_key(sym1):
                lastDate[sym1]=bardate
            if not lastDate.has_key(sym2):
                lastDate[sym2]=bardate    
                
            if lastDate[sym1] < timestamp and lastDate[sym2] < timestamp:
                logging.info("Processing Bar: %s %s %s %s" %(date, sym1, sym2, bardict[sym1]))
                lastDate[sym1]=bardate
                lastDate[sym2]=bardate
                timestamp=bardate
                                    
                bar1=astrat.getBar(bardict[sym1][sym1], sym1, int(timestamp), bardict[sym1][sym1+'_High'], bardict[sym1][sym1+'_Low'], bardict[sym1][sym1+'_Volume'])
                bar2=astrat.getBar(bardict[sym2][sym2], sym2, int(timestamp), bardict[sym2][sym2+'_High'], bardict[sym2][sym2+'_Low'], bardict[sym2][sym2+'_Volume'])
                signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                proc_signals(signals, params, symPair, timestamp)
            
        except Exception as e:
            logging.error("proc_onBar", exc_info=True)

                
def get_entryState():
    global pairs
    global pos
    
    for [file, sym, param] in pairs:
        try:
            (sysqty, syscur, sysexch, system)=param
            
            signal=get_model_sig(system)
            if len(signal.index) > 0:
                signal=signal.iloc[-1]
                #print signal['comment']
                jsondata = json.loads(signal['comment'])
                entryState=jsondata['Entry']
                exitState=jsondata['Exit']
                symPair=jsondata['symPair']
                logging.info('SymPair: ' + symPair + ' System: ' + system + ' Entry: ' + str(entryState) + ' Exit: ' + str(exitState))
                if not pos.has_key(symPair):
                    pos[symPair]=dict()
                pos[symPair][sym]=signal['signals'] * signal['safef']
                logging.info("Initializing " + sym + ' with position: ' + str(pos[symPair][sym]))              
                astrat.updateEntry(symPair, entryState, exitState)
        except Exception as e:
            logging.error("get_entryState", exc_info=True)


def proc_signals(signals, params, symPair, timestamp):
    global pos
    global totalpos
    
    if not pos.has_key(symPair):
            pos[symPair]=dict()
            
    if signals and len(signals) >= 1:
        for signal in signals:
            (barSym, barSig, barCmt)=signal
            logging.info("Processing Signal: " + barSym + '_' + barCmt)
            if pos[symPair].has_key(barSym):
                pos[symPair][barSym]=pos[symPair][barSym] + barSig
            else:
                pos[symPair][barSym]=barSig
                
            if totalpos.has_key(barSym):
                totalpos[barSym]=totalpos[barSym] + barSig
            else:
                totalpos[barSym]=barSig
            
            (sysqty, syscur, sysexch, sysfile)=params[barSym]
            generate_model_sig(sysfile, str(timestamp), totalpos[barSym], 1, barCmt)
           
            if totalpos[barSym] == 0:
                totalpos.pop(barSym, None)
                
            if pos[symPair][barSym] == 0:
                pos[symPair].pop(barSym, None)


def start_bar():
    global SST
    global pairs
    tickers=np.array(pairs,dtype=object)[:,1]
    get_last_bars(tickers, 'Close', onBar)


def onBar(bar, symbols):
    try:
        global SST
        global pairs
        bar=bar.set_index('Date')
        
        SST = bar
        #SST.combine_first(bar).sort_index()
        #SST = SST.fillna(method='pad')
        for sym in symbols:
            logging.info("onBar: " + sym)
            print 'onBar',sym
        #Proc
        seen=dict()
        #if len(symbols) > 0:
        for [file1, sym1, param1] in pairs:
            for [file2, sym2, param2] in pairs:
                if not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
        		 seen[sym1+sym2]=1
        		 seen[sym2+sym1]=1
        		 (sym1,sym2,mult1,mult2)=pparams[sym1+sym2]
        		 proc_onBar(sym1,sym2,mult1,mult2)
        #for sym1 in symbols:
        #    for sym2 in symbols:
        #        if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
        #            seen[sym1+sym2]=1
        #            seen[sym2+sym1]=1
        #            (sym1,sym2,mult1,mult2)=pparams[sym1+sym2]
        #            proc_onBar(sym1,sym2,mult1,mult2)
    except Exception as e:
        logging.error("onBar", exc_info=True)


    
def get_bar(sym):
    global SST
    bar={ 'Date':SST.index.values[-1], sym:SST[sym].values[-1], sym+'_High':SST[sym+'_High'].values[-1],
         sym+'_Low':SST[sym+'_Low'].values[-1], sym+'_Volume':SST[sym+'_Volume'].values[-1] }
    print bar
    return bar

    
def start_prep():
    global pairs
    global SST
    seen=dict()
    #Prep
    threads = []
    for [file1, sym1, mult1] in pairs:
        #print "sym: " + sym1
        for [file2, sym2, mult2] in pairs:
            if not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
                logging.info("Prepping " + sym1 + sym2)
                seen[sym1+sym2]=1
                seen[sym2+sym1]=1
                
                pparams[sym1+sym2]=[sym1,sym2,mult1,mult2]
                pparams[sym2+sym1]=[sym1,sym2,mult1,mult2]
                
                prep_pair(sym1, sym2, mult1, mult2)
                #sig_thread = threading.Thread(target=prep_pair, args=[sym1, sym2, mult1, mult2])
                #sig_thread.daemon=True
                #threads.append(sig_thread)
                
                
    #[t.start() for t in threads]
    #[t.join() for t in threads]
    
    #threads=[]
    #seen=dict()
   
    #Proc
    #for [file1, sym1, param1] in pairs:
    #    #print "sym: " + sym1
    #    for [file2, sym2, param2] in pairs:
    #        if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
    #		logging.info("Processing " + sym1 + sym2)
    #            seen[sym1+sym2]=1
    #            seen[sym2+sym1]=1
    #            sig_thread = threading.Thread(target=proc_pair, args=[sym1, sym2, param1, param2])
    #            sig_thread.daemon=True
    #            threads.append(sig_thread)
    #[t.start() for t in threads]
    
    

pos=dict()
totalpos=dict()
bardict=dict()
lastDate=dict()
SST=get_bar_history(pairs, 'Close')
if SST.shape[0] > 3000:
    SST=SST.tail(3000)
get_entryState()
start_prep()
start_bar()




