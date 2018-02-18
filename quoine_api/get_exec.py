# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from dateutil.parser import parse
import operator
from pandas_datareader import data as pd_data, wb as pd_wb
import re
from dateutil import parser
import datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    from matplotlib.finance import quotes_historical_yahoo
except ImportError:
    from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold
import os
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import sys
from matplotlib.mlab import recs_join
from django.forms.utils import from_current_timezone
from coin import *
import    sys
import    os
import os
from    main.elasticmodels    import    *
from    datetime    import    datetime    
import    subprocess
import    logging
import    psycopg2
import    pandas    as    pd
import    contextlib
import    itertools
from    math    import    sqrt
from    operator    import    add
import    sys
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import quandl
import matplotlib.pyplot as plt
import numpy as np
import elasticsearch
import pyelasticsearch
from pyelasticsearch import bulk_chunks
from datetime import datetime
from dateutil.parser import parse
import time
import requests
import urllib
import urllib2
import json

import time
import requests
import urllib
import urllib2
import json
import jwt
idx_name='forecast'


token=''

def get_token(path):
    global token
    
    token_id = ''
    user_secret = ''
    
    auth_payload = {
      'path': path,
      'nonce': get_nonce_id(),
      'token_id': token_id
    }
    signature = jwt.encode(auth_payload, user_secret, algorithm='HS256')

    return signature    


def get_quoine_portfolio_qty(symbol):
    '''
    # GET /accounts/balance
    [
    {
        "currency": "BTC",
        "balance": "0.04925688"
    },
    {
        "currency": "USD",
        "balance": "7.17696"
    },
    {
        "currency": "JPY",
        "balance": "356.01377"
    }
    ]
    '''
    path='/accounts/balance'
    
    access_token=get_token(path)
    
    headers={ 'X-Quoine-API-Version': '2',
              'X-Quoine-Auth': access_token,
              'Content-Type': 'application/json'
    }
    
    # make a string with the request type in it:
    url = 'https://api.quoine.com' + path
    r = requests.get(url, headers=headers, allow_redirects=True)
    bals=json.loads(r.content)
    qty=0
    try:
        for bal in bals:
            if bal['currency'] == 'BTC':
                qty += float(bal['balance'])
                
    except Exception as e:
        print (e)
    return qty

def get_quoine_open_order_qty(symbol):
    '''
    # GET /orders?funding_currency=:currency&product_id=:product_id&status=:status&with_details=1
    {
  "models": [
    {
      "id": 2157474,
      "order_type": "limit",
      "quantity": "0.01",
      "disc_quantity": "0.0",
      "iceberg_total_quantity": "0.0",
      "side": "sell",
      "filled_quantity": "0.0",
      "price": "500.0",
      "created_at": 1462123639,
      "updated_at": 1462123639,
      "status": "live",
      "leverage_level": 1,
      "source_exchange": "QUOINE",
      "product_id": 1,
      "product_code": "CASH",
      "funding_currency": "USD",
      "currency_pair_code": "BTCUSD",
      "order_fee": "0.0",
      *
      "executions": []
      *
    }
  ],
  "current_page": 1,
  "total_pages": 1
}
    '''
    path='/orders?product_id=5&status=live&with_details=1'
    
    access_token=get_token(path)
    
    headers={ 'X-Quoine-API-Version': '2',
              'X-Quoine-Auth': access_token,
              'Content-Type': 'application/json'
    }
    
    # make a string with the request type in it:
    url = 'https://api.quoine.com' + path
    r = requests.get(url, headers=headers, allow_redirects=True)
    orders=json.loads(r.content)
    qty=0
    try:
        for order in orders['models']:
            if order['currency_pair_code'] == 'BTCJPY':
                if order['side'] == 'sell':
                    qty -= float(order['quantity'])
                elif order['side'] == 'buy':
                    qty += float(order['quantity'])
    except Exception as e:
        print (e)
    return qty

def get_quoine_pos(symbol):
    portf_qty=get_quoine_portfolio_qty(symbol)
    order_qty=get_quoine_open_order_qty(symbol)
    qty=portf_qty + order_qty
    return qty