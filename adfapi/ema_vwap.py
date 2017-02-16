import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import sys
from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol
import talib as ta
import logging
import    os
sys.path.append("../")
import paper.calc as calc
from statsmodels.regression.linear_model import OLS
#from numba import jit

##### Do not change this function definition ####
pairSeries=dict()
pairHSeries=dict()
pairLSeries=dict()
pairVSeries=dict()
pairOSeries=dict()
tsPairratio=dict()
tsZscore=dict()
tsDates=dict()
indSmaZscore=dict()
intDatapoints=1000
sentEntryOrder = dict()
sentExitOrder = dict()
entryOrderPrice =dict()
exitOrderPrice=dict() 
crossAbove=dict()
crossBelow=dict()
intSMALength = 30
instPair1Factor=1
instPair2Factor=1
dblQty=1
dblQty2=1

dblUpperThreshold = 1;
dblLowerThreshold = 0;
dblRiskPer = 0.05;
dblTargetPer = 0.05;
dblStartStop = 0.06;
dblPlaceStop = 0.14;
dblBBUOrder = 2;
dblBBLOrder = 2;

def procBar(bar1, bar2, pos, trade):
    global pairSeries
    global pairHSeries
    global pairLSeries
    global pairVSeries
    global tsPairratio
    global tsZscore
    global tsDates
    global indSmaZscore
    global intDatapoints
    global sentEntryOrder 
    global sentExitOrder 
    global entryOrderPrice 
    global exitOrderPrice 
    global intSMALength 
    global dblUpperThreshold 
    global dblLowerThreshold 
    global instPair1Factor
    global instPair2Factor
    global dblQty
    global dblQty2 
    global crossAbove
    global crossBelow
    
    #logging.info('procBar: %s %s %s' % (bar1, pos, trade))
    
    if bar1['Close'] > 0 and bar2['Close'] > 0:
        xd = bar1['Close'] * instPair1Factor
        yd = bar2['Close'] * instPair2Factor
        sym1=bar1['Symbol']
        sym2=bar2['Symbol']
        #logging.info("s106:procBar:" + sym1 + "," + sym2)
        if not pairSeries.has_key(sym1):
            pairSeries[sym1]=list()
            pairHSeries[sym1]=list()
            pairLSeries[sym1]=list()
            pairVSeries[sym1]=list()
        if not pairSeries.has_key(sym2):
            pairSeries[sym2]=list()
            pairHSeries[sym2]=list()
            pairLSeries[sym2]=list()
            pairVSeries[sym2]=list()
        if not tsPairratio.has_key(sym1+sym2):
            tsPairratio[sym1+sym2]=list()
        if not tsPairratio.has_key(sym2+sym1):
            tsPairratio[sym2+sym1]=list()
        if not tsZscore.has_key(sym1+sym2):
            tsZscore[sym1+sym2]=list()
        if not tsZscore.has_key(sym2+sym1):
            tsZscore[sym2+sym1]=list()
        if not tsDates.has_key(sym1):
            tsDates[sym1]=list()
        if not tsDates.has_key(sym2):
            tsDates[sym2]=list()
        if not tsDates.has_key(sym1+sym2):
            tsDates[sym1+sym2]=list()
        if not tsDates.has_key(sym2+sym1):
            tsDates[sym2+sym1]=list()
        if not crossAbove.has_key(sym1+sym2):
            crossAbove[sym1+sym2]=False
        if not crossBelow.has_key(sym1+sym2):
            crossBelow[sym1+sym2]=False
        if not sentEntryOrder.has_key(sym1+sym2):
            sentEntryOrder[sym1+sym2]=False
        if not sentExitOrder.has_key(sym1+sym2):
            sentExitOrder[sym1+sym2]=False
        
        if bar1['Date'] not in tsDates[sym1]:
            pairSeries[sym1].append(bar1['Close'])
            pairHSeries[sym1].append(bar1['High'])
            pairLSeries[sym1].append(bar1['Low'])
            pairVSeries[sym1].append(bar1['Volume'])
            tsDates[sym1].append(bar1['Date'])
        
        if bar2['Date'] not in tsDates[sym2] and sym1 != sym2:
            pairSeries[sym2].append(bar2['Close'])
            pairHSeries[sym2].append(bar2['High'])
            pairLSeries[sym2].append(bar2['Low'])
            pairVSeries[sym2].append(bar2['Volume'])
            tsDates[sym2].append(bar2['Date'])

        dblRatioData = bar1['Close'] / bar2['Close'];
        tsPairratio[sym1+sym2].append( dblRatioData );
        
        if sym1 != sym2:
            dblRatioData2 = bar2['Close'] / bar1['Close'];
            tsPairratio[sym2+sym1].append( dblRatioData2 );

        if len(tsPairratio[sym1+sym2])< intDatapoints or len(tsPairratio[sym2+sym1]) < intDatapoints:
            return []

        iStart = len(tsPairratio[sym1+sym2]) - intDatapoints;
        iEnd = len(tsPairratio[sym1+sym2]) - 1;
        dblAverage = np.mean(tsPairratio[sym1+sym2][iStart:iEnd]);
        dblRatioStdDev = np.std(tsPairratio[sym1+sym2][iStart:iEnd]);
        dblResidualsData = (dblRatioData - dblAverage);
        dblZscoreData = (dblRatioData - dblAverage) / dblRatioStdDev;
        tsZscore[sym1+sym2].append(dblZscoreData)
        tsDates[sym1+sym2].append(bar1['Date'])
        
        if sym1 != sym2:
            iStart2 = len(tsPairratio[sym2+sym1]) - intDatapoints;
            iEnd2 = len(tsPairratio[sym2+sym1]) - 1;
            dblAverage2 = np.mean(tsPairratio[sym2+sym1][iStart2:iEnd2]);
            dblRatioStdDev2 = np.std(tsPairratio[sym2+sym1][iStart2:iEnd2]);
            dblResidualsData2 = (dblRatioData2 - dblAverage2);
            dblZscoreData2 = (dblRatioData2 - dblAverage2) / dblRatioStdDev2;
            tsZscore[sym2+sym1].append(dblZscoreData2)
            tsDates[sym2+sym1].append(bar2['Date'])
         
        signals=pd.DataFrame()
        #signals['Date']=tsDates[sym1+sym2]
        #signals['tsZscore']=tsZscore[sym1+sym2]
        #signals['tsZscore2']=tsZscore[sym2+sym1]
        #signals['indSmaZscore']=pd.rolling_mean(signals['tsZscore'], intSMALength, min_periods=1)
        #signals['indSmaZscore2']=pd.rolling_mean(signals['tsZscore2'], intSMALength, min_periods=1)    
        
        signals['Date']=tsDates[sym1]
        signals['indEMA9']=ta.EMA(np.array(pairSeries[sym1]), timeperiod=9)
        
        
        df=pd.DataFrame({ 'v' : pairVSeries[sym1], 
                          'h' : pairHSeries[sym1], 
                          'l' : pairLSeries[sym1], 
                         },
                         columns=['v','h','l'] )
        
        signals['vwap_pandas'] = (df.v*(df.h+df.l)/2).cumsum() / df.v.cumsum()
        
        v = df.v.values
        h = df.h.values
        l = df.l.values

        
        signals['vwap'] = np.cumsum(v*(h+l)/2) / np.cumsum(v)
        #signals['vwap_numba'] = vwap(v,h,l)
        
        '''
        (signals['indBbu'], signals['indBbm'], signals['indBbl']) = ta.BBANDS(
            np.array(signals['tsZscore']), 
            timeperiod=intSMALength,
            # number of non-biased standard deviations from the mean
            nbdevup=dblBBUOrder,
            nbdevdn=dblBBLOrder,
            # Moving average type: simple moving average here
            matype=0)
            
        (signals['indBbu2'], signals['indBbm2'], signals['indBbl2']) = ta.BBANDS(
            np.array(signals['tsZscore2']), 
            timeperiod=intSMALength,
            # number of non-biased standard deviations from the mean
            nbdevup=dblBBUOrder,
            nbdevdn=dblBBLOrder,
            # Moving average type: simple moving average here
            matype=0)
        
        
        signals['indSmaZscore']=ta.SMA(np.array(signals['tsZscore']), intSMALength)      
        signals['indSmaZscore2']=ta.SMA(np.array(signals['tsZscore2']), intSMALength)        
        '''
        signals=signals.set_index('Date')
               
        if len(signals['indEMA9']) < 5: # or len(signals['indSmaZscore']) < 5 or len(signals['indSmaZscore2']) < 5:
            return [];

        #updateCointData();

        
        if trade:
                #print strOrderComment
                
                #(z1CBBbu, z1CABbu)=crossCheck(signals, 'bb'+sym1+sym2, 'tsZscore', 'indBbu')
                #(z1CBBbl, z1CABbl)=crossCheck(signals, 'bb2'+sym1+sym2, 'tsZscore2','indBbl')
                
                #print ' crossAbove: ' + str(crossAbove) + ' crossBelow: ' + str(crossBelow)
                #crossBelow = signals['tsZscore'].iloc[-1] >= signals['indSmaZscore'].iloc[-1] and crossBelow.any()
                #crossAbove = signals['tsZscore'].iloc[-1] <= signals['indSmaZscore'].iloc[-1] and crossAbove.any()
                #print ' crossAbove: ' + str(crossAbove) + ' crossBelow: ' + str(crossBelow)
    
                (crossBelow[sym1+sym2], crossAbove[sym1+sym2])=crossCheck(signals, sym1+sym2, 'indEMA9', 'vwap')           
                if not sentEntryOrder[sym1+sym2] and not pos.has_key(bar1['Symbol']): # and not pos.has_key(bar2['Symbol']):
                    logging.info('procBar:crossCheck (Below, Above): %s %s' % (crossBelow[sym1+sym2], crossAbove[sym1+sym2]))

                    
                    if crossAbove[sym1+sym2]:
                        
                        sentEntryOrder[sym1+sym2] = True
                        sentExitOrder[sym1+sym2] = False
                        strOrderComment =  '{"Entry": 1, "Exit": 0, "symPair": "' + sym1+sym2 + '", "indEMA9": ' + str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2)) + '}';
                        strOrderComment2 = '{"Entry": 1, "Exit": 0, "symPair": "' + sym1+sym2 + '", "indEMA9": '+ str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2))+  '}';
                        entryOrderPrice[sym1+sym2]=bar1['Close']
                        return ([[bar1['Symbol'], -abs(dblQty), strOrderComment]])
                    elif crossBelow[sym1+sym2]:
                        sentEntryOrder[sym1+sym2] = True
                        sentExitOrder[sym1+sym2] = False
                        strOrderComment =  '{"Entry": 1, "Exit": 0, "symPair": "' + sym1+sym2 + '", "indEMA9": ' + str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2)) + '}';
                        strOrderComment2 = '{"Entry": 1, "Exit": 0, "symPair": "' + sym1+sym2 + '", "indEMA9": '+ str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2))+  '}';
                
                        entryOrderPrice[sym1+sym2]=bar1['Close']
                        return ([[bar1['Symbol'], abs(dblQty), strOrderComment]])
    
                elif not sentExitOrder[sym1+sym2] and pos.has_key(bar1['Symbol']): # and pos.has_key(bar2['Symbol']):
                    logging.info('procBar:crossCheck (Below, Above):  %s %s' % (crossBelow[sym1+sym2], crossAbove[sym1+sym2]))
                    openAt=entryOrderPrice[sym1+sym2]
                    closeAt=bar1['Close']
                    qty=abs(pos[bar1['Symbol']])
                    if pos[bar1['Symbol']] > 0:
                        side='long'
                    else:
                        side='short'
                    mult=100000
                    (pl, value)=calc.calc_pl(openAt, closeAt, qty, mult, side)
                    plval=50
                    print pos[bar1['Symbol']], pl, value, value * 0.01
                    if pos[bar1['Symbol']] < 0 and pl > plval:
                        sentEntryOrder[sym1+sym2] = False;
                        sentExitOrder[sym1+sym2] = True;
                        strOrderComment =  '{"Entry": 0, "Exit": 1, "symPair": "' + sym1+sym2 + '", "indEMA9": ' + str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2)) + '}';
                        strOrderComment2 = '{"Entry": 0, "Exit": 1, "symPair": "' + sym1+sym2 + '", "indEMA9": '+ str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2))+  '}';
                
                        return ([[bar1['Symbol'], -pos[bar1['Symbol']], strOrderComment]])
    
                    elif pos[bar1['Symbol']] > 0 and pl < plval:
                        print pl, value, value * 0.01
                        sentEntryOrder[sym1+sym2] = False;
                        sentExitOrder[sym1+sym2] = True;
                        strOrderComment =  '{"Entry": 0, "Exit": 1, "symPair": "' + sym1+sym2 + '", "indEMA9": ' + str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2)) + '}';
                        strOrderComment2 = '{"Entry": 0, "Exit": 1, "symPair": "' + sym1+sym2 + '", "indEMA9": '+ str(round(signals.iloc[-1]['indEMA9'], 2)) + ', "vwap": ' + str(round(signals.iloc[-1]['vwap'], 2))+  '}';
                
                        return ([[bar1['Symbol'], -pos[bar1['Symbol']], strOrderComment]])
'''               
@jit
def vwap(v, h, l):
    tmp1 = np.zeros_like(v)
    tmp2 = np.zeros_like(v)
    for i in range(0,len(v)):
        tmp1[i] = tmp1[i-1] + v[i] * ( h[i] + l[i] ) / 2.
        tmp2[i] = tmp2[i-1] + v[i]
    return tmp1 / tmp2
'''

def getBar(price, symbol, date, high=0, low=0, vol=0):
    bar=dict()
    bar['Close']=price
    bar['Symbol']=symbol
    bar['Date']=date
    if high:
        bar['High']=high
    if low:
        bar['Low']=low
    if vol:
        bar['Volume']=vol
    return bar

def updateEntry(symPair, entryOrder, exitOrder):
    sentEntryOrder[symPair] = entryOrder
    sentExitOrder[symPair] = exitOrder

def crossCheck(signals, symPair, tsz, check2):
    global crossBelow
    global crossAbove
    if not crossBelow.has_key(symPair):
        crossBelow[symPair]=False
    if not crossAbove.has_key(symPair):
        crossAbove[symPair]=False
        
    if signals.iloc[-2][tsz] > signals.iloc[-2][check2]  \
            and                                                         \
       signals.iloc[-1][tsz] <= signals.iloc[-1][check2]:
            crossBelow[symPair]=False
            crossAbove[symPair]=True
    print 'Prior: ',tsz, signals.iloc[-2][tsz], check2, signals.iloc[-2][check2]
    print 'Current: ',tsz, signals.iloc[-1][tsz], check2, signals.iloc[-1][check2]
    logging.info('Prior crossCheck: %s %s %s %s' %( tsz, signals.iloc[-2][tsz], check2, signals.iloc[-2][check2] ) )
    logging.info('Current crossCheck: %s %s %s %s' %( tsz, signals.iloc[-1][tsz], check2, signals.iloc[-1][check2] ) )
    if signals.iloc[-2][tsz] < signals.iloc[-2][check2] \
           and                                                         \
       signals.iloc[-1][tsz] >= signals.iloc[-1][check2]:
             crossBelow[symPair]=True
             crossAbove[symPair]=False
    
    return (crossBelow[symPair], crossAbove[symPair])
#def updateEntry(systemname, broker, sym1, sym2, currency, date, isLive):
#    data=portfolio.get_portfolio(systemname, broker, date, isLive)
#    qty1=portfolio.get_pos(data, broker, sym1, currency, date)
#    qty2=portfolio.get_pos(data, broker, sym1, currency, date)
    