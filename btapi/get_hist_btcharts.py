import requests
import urllib
import urllib2
import pandas as pd
from io import StringIO
import numpy as np
from btapi.raw_to_ohlc import feed_to_ohlc, feed_ohlc_to_csv, get_feed_ohlc

import logging

def get_hist_btcharts(exchange):
    #url='http://api.bitcoincharts.com/v1/csv/'
    #http://www.quandl.com/markets/bitcoin
    url = 'http://api.bitcoincharts.com/v1/trades.csv'
    values = {'symbol' : exchange} #, 'start' : '1420121275'}
   
    response = requests.get(url, params=values, json=values);
    #print response.text;
    return response.text;

def get_bthist(ticker, exchange):
    #exchange='bitstampUSD'
    try:
        data=get_hist_btcharts(exchange);
        dataSet = pd.read_csv(StringIO(data))
        #dataSet=dataSet.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        for i in dataSet.index:
            row=dataSet.ix[i]
            feed_to_ohlc(ticker, exchange, row[1], row[0], row[2])
        return feed_ohlc_to_csv(ticker, exchange)
    except Exception as e:
        logging.error("get_bthist", exc_info=True)
    
    return get_feed_ohlc(ticker, exchange)

