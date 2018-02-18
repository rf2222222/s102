import numpy as np
import pandas as pd
import time
import os.path

import json
from pandas.io.json import json_normalize
from seitoolz.signal import get_model_pos
from time import gmtime, strftime, localtime, sleep
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import threading
debug=False
lock = threading.Lock()

def get_account_value(systemname, broker, date):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    
    if os.path.isfile(filename):
        with lock:
            dataSet = pd.read_csv(filename, index_col=['Date'])
            if 'PurePL' not in dataSet:
                dataSet['PurePL']=0
            if 'qty_pnl' not in dataSet:
                dataSet['qty_pnl']=0
            if 'trade_qty_pnl' not in dataSet:
                dataSet['trade_qty_pnl']=0
            if 'real_qty_pnl' not in dataSet:
                dataSet['real_qty_pnl']=0
            if 'real_trade_qty_pnl' not in dataSet:
                dataSet['real_trade_qty_pnl']=0
            return dataSet.reset_index().iloc[-1]
    else:
        dataSet=make_new_account(systemname, broker, date)
        return dataSet.reset_index().iloc[-1]

def update_account_value(systemname, broker, account):
    filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
    
    if os.path.isfile(filename):
        with lock:
            dataSet = pd.read_csv(filename, index_col=['Date'])
    else:
        dataSet=make_new_account(systemname, broker,account['Date'])

    dataSet=dataSet.reset_index()
    dataSet=dataSet.append(account)
    dataSet=dataSet.set_index('Date')
    with lock:
        dataSet.to_csv(filename)
    #print 'Account Update: ' + broker + ' Balance: ' + str(account['balance']) + ' PurePNL:' + str(account['PurePL'])
    return account

def update_account_pnl(systemname, broker, tradepl, purepl, buypower, unr_pnl, pure_unr_pnl, date, share_pl=0, real_share_pl=0):
    account=get_account_value(systemname, broker, date)
    account['balance']=account['balance']+tradepl
    account['purebalance']=account['purebalance']+purepl
    account['buy_power']=account['buy_power']+buypower
    account['real_pnl']=account['real_pnl'] + tradepl
    account['PurePL']=account['PurePL'] + purepl
    account['unr_pnl']=unr_pnl
    account['pure_unr_pnl']=pure_unr_pnl
    account['mark_to_mkt']=account['balance']+account['unr_pnl']
    account['pure_mark_to_mkt']=account['purebalance']+account['pure_unr_pnl']
    account['trade_qty_pnl']=share_pl
    account['qty_pnl']=account['qty_pnl'] + share_pl
    account['real_trade_qty_pnl']=real_share_pl
    account['real_qty_pnl']=account['real_qty_pnl'] + real_share_pl
    account['Date']=date
    account=update_account_value(systemname, broker, account)
    return account
    
def make_new_account(systemname, broker, date):
    with lock:
        filename='./data/paper/' + broker + '_' + systemname + '_account.csv'
        dataSet=pd.DataFrame([[date, 'paperUSD',0,0,2000000,0,0,0,'USD',0,0,0,0,0,0,0]], 
            columns=['Date','accountid','balance','purebalance','buy_power','unr_pnl','real_pnl','PurePL','currency',
                     'mark_to_mkt','pure_mark_to_mkt','pure_unr_pnl', 'qty_pnl', 'trade_qty_pnl', 'real_qty_pnl', 'real_trade_qty_pnl'])
        dataSet=dataSet.set_index('Date')
        dataSet.to_csv(filename)
    return dataSet