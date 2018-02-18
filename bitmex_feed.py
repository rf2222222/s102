import sys
import threading
import websocket
import traceback
from time import sleep
import json
import string
import logging
import urlparse
import math
import urllib
import urllib2
import json
import time
import hmac,hashlib
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
from datetime import datetime
from dateutil.parser import parse
import time
import requests
import urllib
import urllib2
import json
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bitmex_auth import APIKeyAuthenticator
import json
import pprint
from dateutil import tz
import calendar

idx_name='forecast'

def actual_kwargs():
    """
    Decorator that provides the wrapped function with an attribute 'actual_kwargs'
    containing just those keyword arguments actually passed in to the function.
    """
    def decorator(function):
        def inner(*args, **kwargs):
            inner.actual_kwargs = kwargs
            return function(*args, **kwargs)
        return inner
    return decorator

import time, urlparse, hmac, hashlib
def generate_nonce():
    return int(round(time.time() * 1000))


# Generates an API signature.
# A signature is HMAC_SHA256(secret, verb + path + nonce + data), hex encoded.
# Verb must be uppercased, url is relative, nonce must be an increasing 64-bit integer
# and the data, if present, must be JSON without whitespace between keys.
#
# For example, in psuedocode (and in real code below):
#
# verb=POST
# url=/api/v1/order
# nonce=1416993995705
# data={"symbol":"XBTZ14","quantity":1,"price":395.01}
# signature = HEX(HMAC_SHA256(secret, 'POST/api/v1/order1416993995705{"symbol":"XBTZ14","quantity":1,"price":395.01}'))
def generate_signature(secret, verb, url, nonce, data):
    """Generate a request signature compatible with BitMEX."""
    # Parse the url so we can remove the base and extract just the path.
    parsedURL = urlparse.urlparse(url)
    path = parsedURL.path
    if parsedURL.query:
        path = path + '?' + parsedURL.query

    # print "Computing HMAC: %s" % verb + path + str(nonce) + data
    message = bytes(verb + path + str(nonce) + data).encode('utf-8')

    signature = hmac.new(secret, message, digestmod=hashlib.sha256).hexdigest()
    return signature

# Naive implementation of connecting to BitMEX websocket for streaming realtime data.
# The Marketmaker still interacts with this as if it were a REST Endpoint, but now it can get
# much more realtime data without polling the hell out of the API.
#
# The Websocket offers a bunch of data as raw properties right on the object.
# On connect, it synchronously asks for a push of all this data then returns.
# Right after, the MM can start using its data. It will be updated in realtime, so the MM can
# poll really often if it wants.
class BitMEXWebsocket():

    # Don't grow a table larger than this amount. Helps cap memory usage.
    MAX_TABLE_LEN = 200

    # We use the actual_kwargs decorator to get all kwargs sent to this method so we can easily pass
    # it to a validator function.
    @actual_kwargs()
    def __init__(self, endpoint=None, symbol=None, api_key=None, api_secret=None, data=None):
        '''Connect to the websocket and initialize data stores.'''
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing WebSocket.")
        self.blockData=data
        
        self.es = elasticsearch.Elasticsearch(port=int(es_port))
        self.frequency=60
                            
        
        self.__validate(self.__init__.actual_kwargs)
        self.__reset(self.__init__.actual_kwargs)

        # We can subscribe right in the connection querystring, so let's build that.
        # Subscribe to all pertinent endpoints
        self.wsURL = self.__get_url()
        self.symbol=symbol
        self.logger.info("Connecting to %s" % self.wsURL)
        self.__connect(self.wsURL, self.symbol)
        self.logger.info('Connected to WS.')

        # Connected. Wait for partials
        self.__wait_for_symbol(symbol)
        if api_key:
            self.__wait_for_account()
        self.logger.info('Got all market data. Starting.')

    def exit(self):
        '''Call this to exit - will close websocket.'''
        self.exited = True
        self.ws.close()

    def proc_rows(self,rows):
        date_now=''
        
        for line in rows:    
            if line['high'] and line['low'] and line['close'] and line['open']:
                row=dict()
                row['timestamp']=line['timestamp']
                row['low']=float(line['low'])
                row['high']=float(line['high'])
                row['open']=float(line['open'])
                row['close']=float(line['close'])
                row['wap']=float(line['close'])
                row['amount']=float(line['volume'])
                volume=row['amount']
                high_price=row['high']
                low_price=row['low']
                close_price=row['close']
                open_price=row['open']
                instrument_id=line['instrument_id']
                frequency=line['frequency']
                
                timestamp=row['timestamp']
                date=line['date']
                date_str=date
                
                print (timestamp, date_str, row['timestamp'])
                if date_str and open_price and high_price and low_price and close_price:
                            mydoc='main.feed.' + str(instrument_id) + '|' + str(timestamp) + '|' + str(frequency)
                            data={
                                    'instrument_id':int(instrument_id),
                                    'frequency':self.frequency,
                                    'date':date_str,
                                    'open':open_price,
                                    'high':high_price,
                                    'low':low_price,
                                    'close':close_price,
                                    'volume':volume,
                                    'wap':close_price,
                                    'timestamp':timestamp
                            }
                            data['my_id']=mydoc
                            data['django_ct']='main.feed'
                            
                            print (mydoc, data)
                            self.es.update(body={"doc": data, "doc_as_upsert" : True },
                                       id=mydoc, 
                                       index=idx_name,
                                       doc_type='feed')
                            
    def get_instrument(self):
        '''Get the raw instrument data for this symbol.'''
        # Turn the 'tickSize' into 'tickLog' for use in rounding
        instrument = self.data['instrument'][0]
        instrument['tickLog'] = int(math.fabs(math.log10(instrument['tickSize'])))
        return instrument

    def get_ticker(self):
        '''Return a ticker object. Generated from quote and trade.'''
        lastQuote = self.data['quote'][-1]
        lastTrade = self.data['trade'][-1]
        ticker = {
            "last": lastTrade['price'],
            "buy": lastQuote['bidPrice'],
            "sell": lastQuote['askPrice'],
            "mid": (float(lastQuote['bidPrice'] or 0) + float(lastQuote['askPrice'] or 0)) / 2
        }

        # The instrument has a tickSize. Use it to round values.
        instrument = self.data['instrument'][0]
        return {k: round(float(v or 0), instrument['tickLog']) for k, v in ticker.iteritems()}

    def funds(self):
        '''Get your margin details.'''
        return self.data['margin'][0]

    def market_depth(self):
        '''Get market depth (orderbook). Returns all levels.'''
        return self.data['orderBookL2']

    def open_orders(self, clOrdIDPrefix):
        '''Get all your open orders.'''
        orders = self.data['order']
        # Filter to only open orders (leavesQty > 0) and those that we actually placed
        return [o for o in orders if str(o['clOrdID']).startswith(clOrdIDPrefix) and o['leavesQty'] > 0]

    def recent_trades(self):
        '''Get recent trades.'''
        return self.data['trade']

    #
    # End Public Methods
    #

    def __connect(self, wsURL, symbol):
        '''Connect to the websocket in a thread.'''
        self.logger.debug("Starting thread")

        self.ws = websocket.WebSocketApp(wsURL,
                                         on_message=self.__on_message,
                                         on_close=self.__on_close,
                                         on_open=self.__on_open,
                                         on_error=self.__on_error,
                                         header=self.__get_auth())

        self.wst = threading.Thread(target=lambda: self.ws.run_forever())
        self.wst.daemon = True
        self.wst.start()
        self.logger.debug("Started thread")

        # Wait for connect before continuing
        conn_timeout = 5
        while not self.ws.sock or not self.ws.sock.connected and conn_timeout:
            sleep(1)
            conn_timeout -= 1
        if not conn_timeout:
            self.logger.error("Couldn't connect to WS! Exiting.")
            self.exit()
            sys.exit(1)

    def __get_auth(self):
        '''Return auth headers. Will use API Keys if present in settings.'''
        if self.config['api_key']:
            self.logger.info("Authenticating with API Key.")
            # To auth to the WS using an API key, we generate a signature of a nonce and
            # the WS API endpoint.
            nonce = generate_nonce()
            return [
                "api-nonce: " + str(nonce),
                "api-signature: " + generate_signature(self.config['api_secret'], 'GET', '/realtime', nonce, ''),
                "api-key:" + self.config['api_key']
            ]
        else:
            self.logger.info("Not authenticating.")
            return []

    def __get_url(self):
        try:
            '''
            Generate a connection URL. We can define subscriptions right in the querystring.
            Most subscription topics are scoped by the symbol we're listening to.
            '''
    
            # You can sub to orderBookL2 for all levels, or orderBook10 for top 10 levels & save bandwidth
            symbolSubs = ["quoteBin1m", "tradeBin1m", "trade", "quote"]
            genericSubs = ["margin"]
    
            subscriptions = [sub + ':' + self.config['symbol'] for sub in symbolSubs]
            #subscriptions += genericSubs
    
            urlParts = list(urlparse.urlparse(self.config['endpoint']))
            urlParts[0] = urlParts[0].replace('http', 'ws')
            urlParts[2] = "/realtime?subscribe=" + string.join(subscriptions, ",")
            return urlparse.urlunparse(urlParts)
        except Exception as e:
            print (e)
            
    def __wait_for_account(self):
        '''On subscribe, this data will come down. Wait for it.'''
        # Wait for the keys to show up from the ws
        while not {'margin', 'position', 'order', 'orderBookL2'} <= set(self.data):
            sleep(0.1)

    def __wait_for_symbol(self, symbol):
        '''On subscribe, this data will come down. Wait for it.'''
        while not {'instrument', 'trade', 'quote','tradeBin1m','quoetBin1m'} <= set(self.data):
            sleep(0.1)

    def __send_command(self, command, args=[]):
        '''Send a raw command.'''
        self.ws.send(json.dumps({"op": command, "args": args}))

    def __on_message(self, ws, message):
        try:
            '''Handler for parsing WS messages.'''
            message = json.loads(message)
            self.logger.debug(json.dumps(message))
    
            table = message['table'] if 'table' in message else None
            action = message['action'] if 'action' in message else None

            if 'subscribe' in message:
                self.logger.debug("Subscribed to %s." % message['subscribe'])
            elif action:
                    
                if table not in self.data:
                    self.data[table] = []

                if table == 'tradeBin1m' or table == 'quoteBin1m':
                        print message['data']
                    
                # There are four possible actions from the WS:
                # 'partial' - full table image
                # 'insert'  - new row
                # 'update'  - update row
                # 'delete'  - delete row
                if action == 'partial':
                    self.logger.debug("%s: partial" % table)
                    #self.data[table] += message['data']
                    # Keys are communicated on partials to let you know how to uniquely identify
                    # an item. We use it for updates.
                    #self.keys[table] = message['keys']
                elif action == 'insert':
                    self.logger.debug('%s: inserting %s' % (table, message['data']))
                    #self.data[table] += message['data']
                    if table == 'trade' or table == 'quote':
                        for row in message['data']:
                            if table == "quote":
                                #[{u'askSize': 7525, u'timestamp': u'2018-01-23T18:36:34.241Z', 
                                #u'symbol': u'XBTM18', u'bidPrice': 11728, u'bidSize': 20000, u'askPrice': 11752}]
                                row['price']=(row['askPrice'] + row['bidPrice']) / 2
                                row['homeNotional']=0
                                row['volume']=0
                                row['amount']=0
                                #print row
                        
                            rows=self.blockData.proc_data(row)
                            if len(rows) > 0:
                                self.proc_rows(rows)
                                print rows
                    # Limit the max length of the table to avoid excessive memory usage.
                    # Don't trim orders because we'll lose valuable state if we do.
                    #if table not in ['order', 'orderBookL2'] and len(self.data[table]) > BitMEXWebsocket.MAX_TABLE_LEN:
                    #    self.data[table] = self.data[table][(BitMEXWebsocket.MAX_TABLE_LEN / 2):]

                elif action == 'update':
                    self.logger.debug('%s: updating %s' % (table, message['data']))
                    # Locate the item in the collection and update it.
                    #for updateData in message['data']:
                    #    item = findItemByKeys(self.keys[table], self.data[table], updateData)
                    #    if not item:
                    #        return  # No item found to update. Could happen before push
                    #    item.update(updateData)
                    #    # Remove cancelled / filled orders
                    #    if table == 'order' and item['leavesQty'] <= 0:
                    #        self.data[table].remove(item)
                elif action == 'delete':
                    self.logger.debug('%s: deleting %s' % (table, message['data']))
                    # Locate the item in the collection and remove it.
                    #for deleteData in message['data']:
                    #    item = findItemByKeys(self.keys[table], self.data[table], deleteData)
                    #    self.data[table].remove(item)
                #else:
                #    raise Exception("Unknown action: %s" % action)
        except:
            self.logger.error(traceback.format_exc())

    def __on_error(self, ws, error):
        '''Called on fatal websocket errors. We exit on these.'''
        if not self.exited:
            self.logger.error("Error : %s" % error)
        self.__connect(self.wsURL, self.symbol)

    def __on_open(self, ws):
        '''Called when the WS opens.'''
        self.logger.debug("Websocket Opened.")

    def __on_close(self, ws):
        '''Called on websocket close.'''
        self.logger.info('Websocket Closed')

    def __validate(self, kwargs):
        '''Simple method that ensure the user sent the right args to the method.'''
        if 'symbol' not in kwargs:
            self.logger.error("A symbol must be provided to BitMEXWebsocket()")
        if 'endpoint' not in kwargs:
            self.logger.error("An endpoint (BitMEX URL) must be provided to BitMEXWebsocket()")
        if 'api_key' not in kwargs:
            self.logger.error("No authentication provided! Unable to connect.")

    def __reset(self, kwargs):
        '''Resets internal datastores.'''
        self.data = {}
        self.keys = {}
        self.config = kwargs
        self.exited = False


# Utility method for finding an item in the store.
# When an update comes through on the websocket, we need to figure out which item in the array it is
# in order to match that item.
#
# Helpfully, on a data push (or on an HTTP hit to /api/v1/schema), we have a "keys" array. These are the
# fields we can use to uniquely identify an item. Sometimes there is more than one, so we iterate through all
# provided keys.
def findItemByKeys(keys, table, matchData):
    for item in table:
        matched = True
        for key in keys:
            if item[key] != matchData[key]:
                matched = False
        if matched:
            return item
        
class BlockchainData:
    def __init__(self, instruments):
        self.data={}
        self.symToInst={}
        for inst in instruments:
            self.symToInst[inst.sym]=inst.id

    def proc_data(self, row):
        sym=row['symbol']
        if not self.data.has_key(sym):
            self.data[sym]={}
            self.data[sym]['frequency']=60
            self.data[sym]['open_price']=0
            self.data[sym]['volume']=0
            self.data[sym]['high_price']=0
            self.data[sym]['low_price']=0
            self.data[sym]['close_price']=0
            self.data[sym]['date_now']=''

        rows=[]        
        from_zone = tz.tzutc()
        row['timestamp']=time.mktime(parse(re.sub('T',' ', row['timestamp'])).timetuple())
        date=datetime.fromtimestamp(row['timestamp'])
        row['amount']=float(row['homeNotional'])
        timestamp=row['timestamp']
        price=float(row['price'])
        
        row['timestamp']=float(row['timestamp'])
        row['price']=float(row['price'])
        timestamp=row['timestamp']
        price=float(row['price'])
    
        date=datetime.fromtimestamp(row['timestamp'])
        
        if self.data[sym]['frequency'] == 1:
            date_str=(date + relativedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%S')
        if self.data[sym]['frequency'] == 60:
            date_str=(date + relativedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:00')
        elif self.data[sym]['frequency'] == 3600:
            date_str=(date + relativedelta(hours=1)).strftime('%Y-%m-%dT%H:00:00')
                
        price=float(row['price'])
        
        if date_str != self.data[sym]['date_now']:
            if self.data[sym]['date_now'] and self.data[sym]['open_price'] and self.data[sym]['high_price'] and self.data[sym]['low_price'] and self.data[sym]['close_price']:
                if self.data[sym]['date_now']:
                    timestamp=calendar.timegm(parse(re.sub('T',' ',self.data[sym]['date_now'])).timetuple())
                    db_date=datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S')
                    row_data={
                            'symbol': sym,
                            'instrument_id': int(self.symToInst[sym]),
                            'frequency':self.data[sym]['frequency'],
                            'date': db_date,
                            'open': self.data[sym]['open_price'],
                            'high': self.data[sym]['high_price'],
                            'low': self.data[sym]['low_price'],
                            'close': self.data[sym]['close_price'],
                            'volume': self.data[sym]['volume'],
                            'wap': price,
                            'timestamp':timestamp
                    }
                    rows.append(row_data)
                    print row_data['date']
                
            self.data[sym]['open_price']=price
            self.data[sym]['volume']=float(row['amount'])
            self.data[sym]['high_price']=price
            self.data[sym]['low_price']=price
            self.data[sym]['close_price']=price
            self.data[sym]['date_now']=date_str
        else:
            self.data[sym]['volume']+=float(row['amount'])
    
            if self.data[sym]['open_price'] == 0:
                self.data[sym]['open_price']=price
    
            if self.data[sym]['high_price'] == 0 or self.data[sym]['high_price'] < price:
                self.data[sym]['high_price']=price
                
            if self.data[sym]['low_price'] == 0 or self.data[sym]['low_price'] > price:
                self.data[sym]['low_price'] = price
                
            self.data[sym]['close_price']=price
        return rows
    
def setup_logger():
    # Prints logger info to terminal
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # Change this to DEBUG if you want a lot more info
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger




def getInstrument(sym, exchange, resource_type, commodity_type, name):
    found=False    
    try:
        inst_list=Instrument.search().filter('match_phrase',sym=sym).sort('id').execute()
        if inst_list and len(inst_list) > 0:
            inst=inst_list[0]
            found=True
    except Exception as e:
        print (e)
    if not found:
        inst=Instrument()
        inst.sym=sym.upper()
        inst.exch=exchange
        inst.save()    

    if not inst.resource_id:
        resource_list= Resource.search().filter('term',**{'company_name.raw':name}).sort('id').execute()
        
        if resource_list and len(resource_list) > 0:
            resource=resource_list[0]
        else:
            resource=Resource()
    else:
        resource_list=Resource.search().filter('term',id=inst.resource_id).sort('id').execute()
        
        if resource_list and len(resource_list) > 0:
            resource=resource_list[0]
        else:
            resource=Resource()
            
    resource.company_name=name
    resource.owner_id=35
    resource.ticker=sym
    resource.exchange=exchange
    resource.resource_type=resource_type
    resource.commodity_type=commodity_type
    resource.save()
            
    inst.resource_id=resource.id
    inst.exch=exchange
    inst.save()
    feed_list=Feed.search()
    
    return inst

