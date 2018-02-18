# -*- coding: utf-8 -*-
import time
import pandas as pd
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import os
import sys
import random
import time
from threading import Event
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
reload(sys)
sys.setdefaultencoding('utf-8')
from coin import *

import    sys

import    os
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import os
#idx_name='beginning'
idx_name='forecast'

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
from coinbase.wallet.client import Client
import oauth2client

import logging
#logging.basicConfig(filename='logs/coinbase_pos.log',level=logging.INFO)

def get_coinbase_portfolio_qty(symbol):
    '''
    {
  "pagination": {
    "ending_before": null,
    "starting_after": null,
    "limit": 25,
    "order": "desc",
    "previous_uri": null,
    "next_uri": null
  },
  "data": [
    {
      "id": "58542935-67b5-56e1-a3f9-42686e07fa40",
      "name": "My Vault",
      "primary": false,
      "type": "vault",
      "currency": "BTC",
      "balance": {
        "amount": "4.00000000",
        "currency": "BTC"
      },
      "created_at": "2015-01-31T20:49:02Z",
      "updated_at": "2015-01-31T20:49:02Z",
      "resource": "account",
      "resource_path": "/v2/accounts/58542935-67b5-56e1-a3f9-42686e07fa40",
      "ready": true
    },
    {
      "id": "2bbf394c-193b-5b2a-9155-3b4732659ede",
      "name": "My Wallet",
      "primary": true,
      "type": "wallet",
      "currency": "BTC",
      "balance": {
        "amount": "39.59000000",
        "currency": "BTC"
      },
      "created_at": "2015-01-31T20:49:02Z",
      "updated_at": "2015-01-31T20:49:02Z",
      "resource": "account",
      "resource_path": "/v2/accounts/2bbf394c-193b-5b2a-9155-3b4732659ede"
    }
  ]
}
    '''
    api_key=''
    api_secret=''
    client = Client(api_key, api_secret)
    accounts = client.get_accounts()
    qty=0
    for account in accounts['data']:
        try:
            if account['currency'] == 'BTC':
                bal=float(account['balance']['amount'])
                if bal > 0:
                    qty += bal
                if bal < 0:
                    qty -= bal
                    
        except Exception as e:
            logging.info (e)
        

    return qty   
    

def get_coinbase_open_order_qty(symbol):
    #curl -D - -H "Authorization: Bearer $ACCESS_TOKEN" https://api.korbit.co.kr/v1/user/orders/open?currency_pair=$CURRENCY_PAIR
    
    api_key=''
    api_secret=''
    client = Client(api_key, api_secret)
    accounts = client.get_accounts()
    #logging.info (accounts)
    qty=0

    for account in accounts['data']:
        try:
            if account['currency'] == 'BTC' or account['currency'] == 'USD':
                orders = client.get_buys(account['id'])
                logging.info (orders)
                
                for order in orders['data']:
                    try:
                        if order['status'] == 'created':
                            if order['amount']['currency'] == 'BTC':
                                if order['resource'] == 'sell':
                                    qty-=float(order['amount']['amount'])
                                if order['resource'] == 'buy':
                                    qty+=float(order['amount']['amount'])
                                
                    except Exception as e:
                        logging.info (e)
                        
                orders = client.get_sells(account['id'])
                logging.info (orders)
                for order in orders['data']:
                    try:
                        if order['status'] == 'created':
                            if order['amount']['currency'] == 'BTC':
                                if order['resource'] == 'sell':
                                    qty-=float(order['amount']['amount'])
                                if order['resource'] == 'buy':
                                    qty+=float(order['amount']['amount'])
                                
                    except Exception as e:
                        logging.info (e)
        except Exception as f:
            logging.info (f)

    return qty   
    
def get_coinbase_pos(symbol):
    order_qty=get_coinbase_open_order_qty(symbol)
    portf_qty=get_coinbase_portfolio_qty(symbol)
    qty=portf_qty + order_qty
    return qty
