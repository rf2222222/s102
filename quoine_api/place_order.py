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

import time
import requests
import urllib
import urllib2
import json
import jwt

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


# Place Order
def buyQuoine(qty):
    nonce=get_nonce_id()
    
    #curl -D - -X POST -H "Authorization: Bearer $ACCESS_TOKEN" 
    #-d "currency_pair=$CURRENCY_PAIR&type=$TYPE&price=$PRICE&coin_amount=$COIN_AMOUNT&nonce=$NONCE" 
    
    path='/orders/'
    
    access_token=get_token(path)
    
    headers={ 'X-Quoine-API-Version': '2',
              'X-Quoine-Auth': access_token,
              'Content-Type': 'application/json'
    }
    
    # make a string with the request type in it:
    url = 'https://api.quoine.com' + path

    data = json.dumps({
          "order": {
            "order_type": "market",
            "product_id": '5',
            "side": "buy",
            "quantity": "%s"%(qty)
          }
        })

    print (data)
    # make a string with the request type in it:
    r = requests.post(url, headers=headers, data=data, allow_redirects=True)
    print (r.content)

def sellQuoine(qty):
    nonce=get_nonce_id()
    
    #curl -D - -X POST -H "Authorization: Bearer $ACCESS_TOKEN" 
    #-d "currency_pair=$CURRENCY_PAIR&type=$TYPE&price=$PRICE&coin_amount=$COIN_AMOUNT&nonce=$NONCE" 
    
    path='/orders/'
    
    access_token=get_token(path)
    
    headers={ 'X-Quoine-API-Version': '2',
              'X-Quoine-Auth': access_token,
              'Content-Type': 'application/json'
    }
    
    # make a string with the request type in it:
    url = 'https://api.quoine.com' + path

    data = json.dumps({
            "order" : {
            "order_type":"market",
            "product_id" : 5,
            "side": "sell",
            "quantity": "%s"%(qty)
          }
        })

    print (data)
    # make a string with the request type in it:
    r = requests.post(url, headers=headers, data=data, allow_redirects=True)
    print (r.content)


def place_order(action, sym, qty):
    
    if action == 'BUY':
        buyQuoine(qty)
    if action == 'SELL':
        sellQuoine(qty)
        
        