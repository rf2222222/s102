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
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual, generate_model_sig, get_model_sig
from seitoolz.paper import adj_size
from time import gmtime, strftime, localtime, sleep
import logging
import threading
import adfapi.s108_pt as astrat
import adfapi.s107 as s107

import adfapi.s106 as s106
import adfapi.s102 as s102
import seitoolz.graph as seigraph
import adfapi.adf_helper as adf
import seitoolz.bars as bars
import re

from proc_signal import send_order, get_bitmex_pos_price

systemdata=pd.read_csv('./data/systems/system.csv')
systemdata=systemdata.reset_index()
commissiondata=pd.read_csv('./data/systems/commission.csv')
commissiondata=commissiondata.reset_index()
commissiondata['key']=commissiondata['Symbol']  + commissiondata['Currency'] + commissiondata['Exchange']
commissiondata=commissiondata.set_index('key')
     
start_time = time.time()

debug=False


logging.basicConfig(filename='logs/system_adf.log',level=logging.WARNING)

pairs=[]
pparams=dict()

sysname='stratADF'
sysname_paper='ADF_paper'

if len(sys.argv) > 1 and sys.argv[1] == 'BTC':
    pairs=[
           ['USD', 'BTC_USD', [1,'USD','IDEALPRO', 's105_BTC_USD']],
           ['KRW', 'BTC_KRW', [1,'USD','IDEALPRO', 's105_BTC_KRW']]]
elif len(sys.argv) > 1 and sys.argv[1] == 'JPY':
    pairs=[
           ['USD', 'XBTUSD', [1,'USD','IDEALPRO', 's107_XBTUSD']],
           ['JPY', 'BTCJPY', [1,'USD','IDEALPRO', 's107_BTCJPY']]]
    
    sysname='ADF_XBT_JPY'
elif len(sys.argv) > 1 and sys.argv[1] == 'JPYC':
    pairs=[
           ['USD', 'BTC_USD', [1,'USD','IDEALPRO', 's107_BTC_USD']],
           ['JPY', 'BTCJPY', [1,'USD','IDEALPRO', 's107_BTCJPY']]]
    
    sysname='ADF_XBT_JPY'
elif len(sys.argv) > 1 and sys.argv[1] == 'XBTXBT':
    pairs=[
           ['USD', 'XBTH18', [1,'USD','IDEALPRO', 's108_XBTH18']],
           ['USD', 'XBTM18', [1,'USD','IDEALPRO', 's108_XBTM18']]]
    
    sysname='ADF_XBT_XBT'        
else:
    pairs=[
           ['USD', 'XBTH18', [1,'USD','IDEALPRO', 's108_XBTH18']],
           ['USD', 'XBTM18', [1,'USD','IDEALPRO', 's108_XBTM18']]]
    
    sysname='ADF_XBT_XBT'        


def prep_pair(sym1, sym2, param1, param2):
        symPair=sym1+sym2
        print 'SymPair: ', symPair
        global pos
        global SST


        proc_paper()
        
        if not pos.has_key(symPair):
            pos[symPair]=dict()

        params=dict()
        
        params[sym1]=param1
        params[sym2]=param2
        
        confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
        print ("Coint Confidence: " + str(confidence) + "%")
        
        for ii in SST.index:
            try:
                priceHist=SST.ix[ii]

                #print ii, priceHist['timestamp']
                timestamp=time.mktime(priceHist['timestamp'].timetuple())
                bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                signals=astrat.procBar(bar1, bar2, pos[symPair], True, False)
                #print ('prep timestamp: ', timestamp)
                proc_signals(signals, params, symPair, timestamp)

                proc_paper_trade(sym1, sym2, param1, param2, signals, priceHist, symPair)
                

            except Exception as e:
                 logging.error('prep_pair', exc_info=True)
                

def start_bar(frequency):
    global SST
    global pairs

    #Paper                            
    global sysname_paper
    global pos_paper
    global totalpos_paper


    tickers=np.array(pairs,dtype=object)[:,1]
    datenow=dict()

 

    while 1:
        sleep(1)
        
        SST=bars.get_bar_elastic_feed(pairs, 'close', frequency) 
        #Proc
        seen=dict()
        (file1, sym1, param1)=pairs[0]
        (file2, sym2, param2)=pairs[1]

        symPair=sym1+sym2
        if not datenow.has_key(symPair):
            datenow[symPair]=0
     
        params=dict()
        params[sym1]=param1
        params[sym2]=param2

        try:
            row=SST.iloc[-1:]
            for i in row.index:
                try:
                    priceHist=row.ix[i]
                    
                    timestamp=time.mktime(priceHist['timestamp'].timetuple())
                    bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                    bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                    
                    if datenow[symPair] != timestamp:
                        signals=astrat.procBar(bar1, bar2, pos[symPair], True, True)
                        print ('proc timestamp: ', timestamp, symPair)
                        if signals and len(signals) >= 1:
                            proc_signals(signals, params, symPair, timestamp)    
                            
                            send_order()
                            proc_paper_trade(sym1, sym2, param1, param2, signals, priceHist, symPair)
                        datenow[symPair]=timestamp
                    
                except Exception as e:
                     logging.error('prep_pair', exc_info=True)
            
            #send_order()
            
        except Exception as f:
            print (f)
            logging.error('prep_pair', exc_info=True)
 
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




    
def get_bar(sym):
    global SST
    return SST[sym][-1]

def get_date():
    global SST
    return SST['date'][-1]

    
def start_prep():
    global pairs
    global SST
    seen=dict()
    #Prep
    threads = []
    #for [file1, sym1, mult1] in pairs:
    #print "sym: " + sym1
    #for [file2, sym2, mult2] in pairs:
    (file1, sym1, mult1)=pairs[0]
    (file2, sym2, mult2)=pairs[1]
    #if sym1 != sym2 and not seen.has_key(sym1+sym2) and not seen.has_key(sym2+sym1):
    logging.info("Prepping " + sym1 + sym2)
    #    seen[sym1+sym2]=1
    #    seen[sym2+sym1]=1
    
    print sym1, sym2
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
   

# Paper

def proc_paper():
    global sysname_paper
    frequency=60
    
    
    refresh_paper(sysname_paper)
        
def refresh_paper(sysname):
    files=['./data/paper/c2_' + sysname + '_account.csv','./data/paper/c2_' + sysname + '_trades.csv', \
    './data/paper/ib_'+ sysname + '_portfolio.csv','./data/paper/c2_' + sysname + '_portfolio.csv',  \
    './data/paper/ib_' + sysname + '_account.csv','./data/paper/ib_' + sysname + '_trades.csv']
    for i in files:
        filename=i
        if os.path.isfile(filename):
            os.remove(filename)
            print ('Deleting ' + filename)

def proc_paper_trade(sym1, sym2, param1, param2, signals, priceHist, symPair):
        #Paper                            
        global sysname_paper
        global pos_paper
        global totalpos_paper

        asks=dict()
        bids=dict()
        
        
        
        if not pos_paper.has_key(symPair):
            pos_paper[symPair]=dict()
        params=dict()
       
        params[sym1]=param1
        params[sym2]=param2
        #confidence=adf.getCoint(SST[sym1], sym1, SST[sym2], sym2)
        #print "Coint Confidence: " + str(confidence) + "%"
        try:
                
                asks[sym1]=priceHist[sym1]
                bids[sym1]=priceHist[sym1]
                asks[sym2]=priceHist[sym2]
                bids[sym2]=priceHist[sym2]
                timestamp=time.mktime(priceHist['timestamp'].timetuple())
                #bar1=astrat.getBar(priceHist[sym1], sym1, int(timestamp))
                #bar2=astrat.getBar(priceHist[sym2], sym2, int(timestamp))
                #signals=astrat.procBar(bar1, bar2, pos[symPair], True)
                #print asks[sym1], bids[sym2]
                if signals and len(signals) >= 1:
                    for signal in signals:
                        (barSym, barSig, barCmt)=signal
                        
                        if pos_paper[symPair].has_key(barSym):
                            pos_paper[symPair][barSym]=pos_paper[symPair][barSym] + barSig
                        else:
                            pos_paper[symPair][barSym]=barSig
                            
                        if totalpos_paper.has_key(barSym):
                            totalpos_paper[barSym]=totalpos_paper[barSym] + barSig
                        else:
                            totalpos_paper[barSym]=barSig
                        
                        model=generate_model_manual(barSym, totalpos_paper[barSym], 1)
                        
                        if totalpos_paper[barSym] == 0:
                            totalpos_paper.pop(barSym, None)
                            
                        if pos_paper[symPair][barSym] == 0:
                            pos_paper[symPair].pop(barSym, None)
                            
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
                        #secType='CASH'
                        #ibsym=sym[0:3]
                        
                        pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                        if ask > 0 and bid > 0:
                            print ask, bid
                            date=datetime.datetime.fromtimestamp(
                                int(timestamp)
                            ).strftime("%Y%m%d %H:%M:%S EST")
                            print 'Date: ', date, ' Signal: ' + barSym + '[' + str(barSig) + ']@' + str(ask)
                            adj_size(model, barSym, sysname_paper, pricefeed,   \
                                sysname_paper,sysname_paper,mult,barSym, secType, True, \
                                    mult, ibsym,currency,exchange, secType, True, date)

        except Exception as e:
            logging.error('proc_pair', exc_info=True)


def syncBitMexPos():
    def syncPos():
        while 1:
            prices=get_bitmex_pos_price()
            for sym in prices:
                astrat.updatePortfolio(sym['symbol'], sym['price'], sym['qty'])        
            time.sleep(300)
    threads = []
    sig_thread = threading.Thread(target=syncPos, args=[])
    sig_thread.daemon=True
    threads.append(sig_thread)
    [t.start() for t in threads]
                    
pos=dict()
totalpos=dict()
pos_paper=dict()
totalpos_paper=dict()

frequency=60
SST=bars.get_bar_elastic_history(pairs, 'close', frequency)
#print (SST)
if SST.shape[0] > 25000:
    SST=SST.tail(25000)

get_entryState()
start_prep()
syncBitMexPos()
start_bar(frequency)







