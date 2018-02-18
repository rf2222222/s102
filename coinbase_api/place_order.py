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
#logging.basicConfig(filename='logs/coinbase_order.log',level=logging.INFO)

def buyCoinbase(qty):
    
    api_key=''
    api_secret=''
    client = Client(api_key, api_secret)
    accounts = client.get_accounts()
    account_id=''
    logging.info (accounts)
    for account in accounts['data']:
        try:
            if account['currency'] == 'BTC':
                account_id=account['id']
        except Exception as f:
            logging.info (f)
            
    if account_id:
        logging.warning ('Coinbase Buy: BTC %.15f' % (qty), client.buy(account_id, amount='%.15f'%(qty), currency='BTC'))

def sellCoinbase(qty):
    
    api_key=''
    api_secret=''
    client = Client(api_key, api_secret)
    
    accounts = client.get_accounts()
    account_id=''
    for account in accounts['data']:
        try:
            if account['currency'] == 'BTC':
                account_id=account['id']
        except Exception as f:
            logging.info (f)
    
    if account_id:
        logging.warning ('Coinbase Sell: BTC %.15f' % (qty), client.sell(account_id, amount='%.15f'%(qty), currency='BTC'))

def place_order(action, sym, qty):
    
    #tid=random.randint(1,10000)
    if action == 'BUY':
        buyCoinbase(qty)
    if action == 'SELL':
        sellCoinbase(qty)

    
#place_order("BUY", 1, "EUR", "CASH", "USD", "IDEALPRO");
