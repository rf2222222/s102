import datetime
import pytz

import pandas as pd
import numpy as np
import logging
import threading
lock = threading.RLock()
barlock = threading.RLock()

btchigh=dict()
btclow=dict()
btcclose=dict()
btcopen=dict()
btcdate=dict()
btcvolume=dict()
btcbar=dict()
                
hasdata=dict()

def feed_to_ohlc(ticker, exchange, price, timestamp, vol):
    global btchigh
    global btclow
    global btcclose
    global btcopen
    global btcdate
    global btcvolume
    global btcbar
    global hasdata
    
    #exchange=ticker + exchange
    if exchange not in btchigh:
        btchigh[exchange]=dict()
    high=btchigh[exchange]
    
    if exchange not in btclow:
        btclow[exchange]=dict()
    low=btclow[exchange]
    
    if exchange not in btcclose:
        btcclose[exchange]=dict()
    close=btcclose[exchange]
    
    if exchange not in btcopen:
        btcopen[exchange]=dict()
    open=btcopen[exchange]
    
    if exchange not in btcdate:
        btcdate[exchange]=dict()
        
    if exchange not in btcbar:
        btcbar[exchange]=pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')

    date=btcdate[exchange]
    
    if exchange not in btcvolume:
        btcvolume[exchange]=dict()
    volume=btcvolume[exchange]
    row=[timestamp, price, vol]
    

    data=btcbar[exchange]
    if len(str(timestamp)) > 0 and price > 0 and vol > 0:
        hour=datetime.datetime.fromtimestamp(
            timestamp, tz=pytz.timezone('US/Eastern')
        ).strftime('%Y-%m-%d %H:%M') 
        
        if len(open) > 0 and open.has_key(hour):
            
        	if row[1] > high[hour]:
        		high[hour]=row[1]
        	if row[1] < low[hour]:
        		low[hour]=row[1]
        	close[hour]=row[1]
        	volume[hour]=volume[hour] + row[2]
        else:
        	open[hour]=row[1]
        	high[hour]=row[1]
        	low[hour]=row[1]
        	close[hour]=row[1]	
        	date[hour]=str(hour) + ":00"
        	volume[hour]=row[2]
        
        if hour in data.index:
            quote=data.loc[hour]
            if high[hour] > quote['High']:
                quote['High']=high[hour]
            if low[hour] < quote['Low']:
                quote['Low']=low[hour]
            quote['Close']=close[hour]
            quote['Volume']=quote['Volume'] + volume[hour]
            if quote['Volume'] < 0:
                quote['Volume'] = 0 
            data.loc[hour]=quote
            #print "Update Bar: bar: sym: " + exchange + " date:" + str(hour) + "open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']) 
                    
        else:
                #if len(data.index) > 1:
                #data=data.reset_index()                
                #data=data.sort_values(by='Date')  
                #quote=data.iloc[-1]
                #logging.info("Close Bar: " + exchange + " date:" + str(quote['Date']) + " open: " + str(quote['Open']) + " high:"  + str(quote['High']) + ' low:' + str(quote['Low']) + ' close: ' + str(quote['Close']) + ' volume:' + str(quote['Volume']))
                #data=data.set_index('Date')
                #data=data.sort_index()
                #with lock:
                #    data.to_csv('./data/from_IB/' + ticker + '_' + exchange + '.csv')
                
                #gotbar=pd.DataFrame([[quote['Date'], quote['Open'], quote['High'], quote['Low'], quote['Close'], quote['Volume'], exchange]], columns=['Date','Open','High','Low','Close','Volume','Symbol']).set_index('Date')
                #with barlock:
                #    gotbar.to_csv('./data/bars/' + ticker + '_' + exchange + '.csv')
                #logging.info("New Bar:   " + exchange + " date:" + str(hour) + " open: " + str(open[hour]) + " high:"  + str(high[hour]) + ' low:' + str(low[hour]) + ' close: ' + str(close[hour]) + ' volume:' + str(volume[hour]))
                        
            data=data.reset_index().append(pd.DataFrame([[hour, open[hour], high[hour], low[hour], close[hour], volume[hour]]], columns=['Date','Open','High','Low','Close','Volume'])).set_index('Date')
        btcbar[exchange] = data   
        #print exchange + ' ' + str(hour) + ' ' + str(open[hour]) + ' ' + \
        #                str(high[hour]) + ' ' + str(low[hour]) + ' ' + \
        #                str(close[hour]) + ' ' + str(volume[hour])
             
    btcdate[exchange]=date
    btcopen[exchange]=open
    btchigh[exchange]=high
    btclow[exchange]=low
    btcclose[exchange]=close
    btcvolume[exchange]=volume
    if exchange not in hasdata:
        hasdata[exchange]=True
        print exchange + ' has data ' 
    #dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
    #		      'Low':low.values(), 'Close':close.values(), 'Volume':volume.values()}, columns=['Date','Open','High','Low','Close','Volume'])	
    #dataSet=dataSet.sort_values(by='Date')
    #dataSet=dataSet.set_index('Date')
    #dataSet.to_csv(outfile)
    #return dataSet

def get_feed_ohlc(ticker, exchange):
    global btchigh
    global btclow
    global btcclose
    global btcopen
    global btcdate
    global btcvolume
    global btcbar
    global hasdata
    
    
    if exchange in btcbar:
        #print exchange + " Found "
        #dataSet=pd.DataFrame({'Date':btcdate[exchange].values(), 'Open':btcopen[exchange].values(), 'High':btchigh[exchange].values(),
        #		      'Low':btclow[exchange].values(), 'Close':btcclose[exchange].values(), 'Volume':btcvolume[exchange].values()}, 
        #        columns=['Date','Open','High','Low','Close','Volume'])	
        return btcbar[exchange]
    else:
        #print exchange + " NOT Found "
        dataSet=pd.DataFrame({},columns=['Date','Open','High','Low','Close','Volume'])
        dataSet=dataSet.set_index('Date')
        return dataSet

    
def feed_ohlc_to_csv(ticker, exchange):
    dataSet=get_feed_ohlc(ticker, exchange)
    dataSet=dataSet.sort_index()
    with lock:
        dataSet.to_csv('./data/from_IB/' + ticker + '_' + exchange + '.csv')
    if dataSet.shape[0] > 0:
        quote=dataSet.reset_index().iloc[-1]
        gotbar=pd.DataFrame([[quote['Date'], quote['Open'], quote['High'], quote['Low'], quote['Close'], quote['Volume'], exchange]], columns=['Date','Open','High','Low','Close','Volume','Symbol']).set_index('Date')
        with barlock:
            gotbar.to_csv('./data/bars/' + ticker + '_' + exchange + '.csv')
    return dataSet

    
    
def raw_to_ohlc_from_csv(infile, outfile):
    df=pd.DataFrame()
    with lock:
        df=pd.read_csv(infile)
    return raw_to_ohlc(df, outfile)
    
def raw_to_ohlc(df, outfile):
    high={}
    low={}
    close={}
    open={}
    date={}
    volume={}

    for i in df.index:
        row=df.ix[i]
        if len(str(row[0])) > 0:
            hour=datetime.datetime.fromtimestamp(
                int(row[0]), tz=pytz.timezone('US/Eastern')
            ).strftime('%Y-%m-%d %H') 
            
            if open.has_key(hour):
                
            	if row[1] > high[hour]:
            		high[hour]=row[1]
            	if row[1] < low[hour]:
            		low[hour]=row[1]
            	close[hour]=row[1]
            	volume[hour]=volume[hour] + row[2]
            else:
            	open[hour]=row[1]
            	high[hour]=row[1]
            	low[hour]=row[1]
            	close[hour]=row[1]	
            	date[hour]=str(hour) + ":00"
            	volume[hour]=row[2]
    dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
    		      'Low':low.values(), 'Close':close.values(), 'Volume':volume.values()}, columns=['Date','Open','High','Low','Close','Volume'])	
    dataSet=dataSet.sort_values(by='Date')
    dataSet=dataSet.set_index('Date')
    with lock:
        dataSet.to_csv(outfile)
    return dataSet

def raw_to_ohlc_min_from_csv(infile, outfile):
    df=pd.DataFrame()
    with lock:
        df=pd.read_csv(infile)
    raw_to_ohlc_min(df)

def raw_to_ohlc_min(df, outfile):     
    high={}
    low={}
    close={}
    open={}
    date={}
    volume={}

    for i in df.index:
        row=df.ix[i]
        if np.any(np.isnan(row)) or np.all(np.isfinite(row)):
                pass
        hour=datetime.datetime.fromtimestamp(
            int(row[0]), tz=pytz.timezone('US/Eastern')
        ).strftime('%Y-%m-%d %H:%M')    
        if len(str(hour)) > 0:
                    
            if open.has_key(hour):
                
            	if row[1] > high[hour]:
            		high[hour]=row[1]
            	if row[1] < low[hour]:
            		low[hour]=row[1]
            	close[hour]=row[1]
            	volume[hour]=volume[hour] + row[2]
            else:
            	open[hour]=row[1]
            	high[hour]=row[1]
            	low[hour]=row[1]
            	close[hour]=row[1]	
            	date[hour]=str(hour)
            	volume[hour]=row[2]
    dataSet=pd.DataFrame({'Date':date.values(), 'Open':open.values(), 'High':high.values(),
    		      'Low':low.values(), 'Close':close.values(), 'Volume':volume.values()}, columns=['Date','Open','High','Low','Close','Volume'])	
    
    dataSet=dataSet.sort_values(by='Date')
    dataSet=dataSet.set_index(['Date'])
    with lock:
        dataSet.to_csv(outfile)
    return dataSet
