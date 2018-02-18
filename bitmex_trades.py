#!/usr/bin/env python
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
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bitmex_auth import APIKeyAuthenticator
import json
import pprint
import time
from dateutil.parser import parse
from seitoolz.signal import get_dps_model_pos, get_model_pos, generate_model_manual, generate_model_sig, get_model_sig
from seitoolz.paper import adj_size

HOST = "https://www.bitmex.com/api/v1"
SPEC_URI = "https://www.bitmex.com/api/explorer/swagger.json"
# testnet
HOST = "https://testnet.bitmex.com"
SPEC_URI = HOST + "/api/explorer/swagger.json"
# See full config options at http://bravado.readthedocs.io/en/latest/configuration.html
config = {
  # Don't use models (Python classes) instead of dicts for #/definitions/{models}
  'use_models': False,
  # This library has some issues with nullable fields
  'validate_responses': False,
  # Returns response in 2-tuple of (body, response); if False, will only return body
  'also_return_response': True,
}

bitMEX = SwaggerClient.from_url(
  SPEC_URI,
  config=config)

pp = pprint.PrettyPrinter(indent=2)

#
# Authenticated calls
#
# To do authentication, you must generate an API key.
# Do so at https://testnet.bitmex.com/app/apiKeys

# Testnet
API_KEY = '31ANC3vt5WeKXyRuWN34Oe3L'
API_SECRET = 'lxexIxOC1hAyXbn1ouT4uwitFmubnaCFYLV3HrNoltLaAVGi'
API_KEY2 = '_sbMzUIQSh7oKC7nRTEzXuBC'
API_SECRET2 = 'onIM0geIUTpTnkHY0eWLE7QssINrk6XbBnOyaEWUzvrfARfR'


request_client = RequestsClient()
request_client.authenticator = APIKeyAuthenticator(HOST, API_KEY, API_SECRET)

bitMEXAuthenticated = SwaggerClient.from_url(
  SPEC_URI,
  config=config,
  http_client=request_client)


def get_bitmex_portfolio_qty(symbol):
        
    # Basic authenticated call
    print('\n---A basic Position GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    res, http_response = bitMEXAuthenticated.Position.Position_get(filter=json.dumps({'symbol': symbol})).result()
    #pp.pprint(res)
    qty=0
    for bal in res:
        try:
            if bal['symbol'] == symbol:
                qty += float(bal['simpleQty'])
        except Exception as e:
            logging.info (e)
    return round(qty,3)

    

def get_bitmex_open_order_qty(symbol):
    # Basic authenticated call
    print('\n---A basic open orders GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    res, http_response = bitMEXAuthenticated.Order.Order_getOrders(filter=json.dumps({'symbol': symbol, 'open': True})).result()
    #pp.pprint(res)
    qty=0
    for order in res:
        try:
            if order['side'] == 'Buy':
                qty+=float(order['simpleOrderQty'])
            if order['side'] == 'Sell':
                qty-=float(order['simpleOrderQty'])
        except Exception as e:
            logging.info (e)
    return round(qty,3)
        
def get_bitmex_pos(symbol):
    order_qty=get_bitmex_open_order_qty(symbol)
    portf_qty=get_bitmex_portfolio_qty(symbol)
    qty=portf_qty + order_qty
    return qty


# Place Order
def buyBitmex(sym, qty):
    res, http_response = bitMEXAuthenticated.Quote.Quote_get(symbol=sym, reverse=True).result()
    #print(res)
    for quote in res:
        if quote['askPrice']:
            price=quote['askPrice']
            orderQty=round(price * qty)
            print ('Ask: ', quote['askPrice'], 'qty: ', qty, 'orderQty:', orderQty)
            # Basic order placement
            # print(dir(bitMEXAuthenticated.Order))
            res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Buy', orderQty=orderQty).result()
            #print(res)
            return True

def sellBitmex(sym, qty):
    res, http_response = bitMEXAuthenticated.Quote.Quote_get(symbol=sym, reverse=True).result()
    #print(res)
    for quote in res:
        if quote['bidPrice']:
            price=quote['bidPrice']
            orderQty=round(price * qty)
            print ('Bid: ', quote['bidPrice'], 'qty: ', qty, 'orderQty:', orderQty)
            # Basic order placement
            # print(dir(bitMEXAuthenticated.Order))
            res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Sell', orderQty=orderQty).result()
            #print(res)
            return True


def place_order(action, sym, qty):
    
    if action == 'BUY':
        buyBitmex(sym, qty)
    if action == 'SELL':
        sellBitmex(sym, qty)
        

def get_trade_hist():
        
    # Basic authenticated call
    print('\n---A basic Position GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    #res, http_response = bitMEXAuthenticated.User.User_getWalletHistory().result()
    #res, http_response = bitMEXAuthenticated.Position.Position_get(filter=json.dumps({'symbol': 'XBTUSD'})).result()
    res, http_response = bitMEXAuthenticated.Execution.Execution_getTradeHistory(reverse=True, count=500).result()
    
    #pp.pprint(res)
    qty=0
    print res
    res=reversed(res)
    
    pos=dict()
    totalpos=dict()
    sysname_paper='Bitmex_History'  
    pos_paper=dict()
    totalpos_paper=dict()
                           
    datenow=dict()
    frequency=60
    
    
    refresh_paper(sysname_paper)
                
    for t in res:
        try:
            if t['execType'] == 'Trade':
                if t['ordStatus'] == 'Filled':
                    print t['transactTime'], t['ordStatus'], t['side'], t['simpleCumQty'], t['symbol'], '@', t['avgPx'], t['currency'], 'Commission: ', t['commission'], round(t['commission'] *  t['avgPx'], 2)
                    
                    currency='USD'
                    barSym=t['symbol']
                    if t['side'].upper() == 'BUY':
                        barSig=1
                    else:
                        barSig=-1
                    barCmt=""
                    barQty=float(t['simpleCumQty'])
                    ask=float(t['avgPx'])
                    bid=float(t['avgPx'])
                    timestamp=time.mktime(t['transactTime'].timetuple())
                    commission_pct=0
                    commission_cash=round(t['commission'] *  t['avgPx'] * barQty, 2)
                    exchange='Bitmex'
                    
                    (sysname_paper, pos_paper, totalpos_paper)=proc_paper_trade(sysname_paper, pos_paper, totalpos_paper, barSym, barSig, barCmt, barQty, ask, bid, timestamp, commission_pct, commission_cash, currency, exchange)
                    
        except Exception as e:
            print (e)
                

def main():
    #pos=get_bitmex_pos('XBTH18')
    #pos2=get_bitmex_pos('XBTM18')
    #print ('Position: ', pos, pos2)
    #pos=round(pos, 3)
    #if pos:
    #print (place_order('BUY','XBTH18',-pos))
    #print (place_order('BUY','XBTM18',-pos2))
    
    #pos=get_bitmex_pos('XBTUSD')
    #pos=round(pos,3)
    #print ('New Position: ', pos)
    get_trade_hist()


        
def refresh_paper(sysname):
    files=['./data/paper/c2_' + sysname + '_account.csv','./data/paper/c2_' + sysname + '_trades.csv', \
    './data/paper/ib_'+ sysname + '_portfolio.csv','./data/paper/c2_' + sysname + '_portfolio.csv',  \
    './data/paper/ib_' + sysname + '_account.csv','./data/paper/ib_' + sysname + '_trades.csv']
    for i in files:
        filename=i
        if os.path.isfile(filename):
            os.remove(filename)
            print ('Deleting ' + filename)

def proc_paper_trade(sysname_paper, pos_paper, totalpos_paper, barSym, barSig, barCmt, barQty, ask, bid, timestamp, commission_pct, commission_cash, currency, exchange):

        
        symPair=barSym
        if not pos_paper.has_key(symPair):
            pos_paper[symPair]=dict()

        try:                                
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
                                        
                sym=barSym
                ibsym=barSym
                secType='BITCOIN'
                ibsym=sym
                
                pricefeed=pd.DataFrame([[ask, bid, 1, 1, exchange, secType, commission_pct, commission_cash]], columns=['Ask','Bid','C2Mult','IBMult','Exchange','Type','Commission_Pct','Commission_Cash'])
                if ask > 0 and bid > 0:
                    print ask, bid
                    date=datetime.datetime.fromtimestamp(
                        int(timestamp)
                    ).strftime("%Y%m%d %H:%M:%S")
                    print 'Date: ', date, ' Signal: ' + barSym + '[' + str(barSig) + ']@' + str(ask)
                    adj_size(model, barSym, sysname_paper, pricefeed,   \
                        sysname_paper,sysname_paper,barQty,barSym, secType, True, \
                            barQty, ibsym,currency,exchange, secType, True, date)
                
                return (sysname_paper, pos_paper, totalpos_paper)
                
        except Exception as e:
            print (e)

    
if    __name__    ==    "__main__":
    try:
        main()
    except    KeyboardInterrupt:
        pass


