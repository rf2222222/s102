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
import os
from os import listdir
from os.path import isfile, join
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
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bitmex_auth import APIKeyAuthenticator
import json
import pprint
from dateutil import tz
import calendar

idx_name='forecast'

def get_blockchain(sym, period, count, iterations):
    binSize='1m'
    if period == 60:
        binSize='1m'
    elif period == 3600:
        binSize='1h'
    # binSize='1d'
    #res, http_response = bitMEXAuthenticated.Trade.Trade_getBucketed(filter=json.dumps(data )).result()
    data=[]   
    url = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=' + binSize + '&partial=false&symbol=' + sym
    url += '&count=' + str(count) + '&start=0&reverse=true'
    r = requests.get(url, allow_redirects=True)
    def proc_data(content):
        rows=[]
        res=json.loads(r.content)
        for row in res:
            try:
                #print row
                from_zone = tz.tzutc()
                row['timestamp']=calendar.timegm(parse(re.sub('T',' ', row['timestamp'])).timetuple())
                date=datetime.fromtimestamp(row['timestamp'])
                #print (row['timestamp'], date)
                rows.append(row)
            except Exception as e:
                print (e)
        return rows
    res=proc_data(r.content)

    data += res
    iterations-=1
    start=0
    while iterations > 0:
        start+=count
        url = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=' + binSize + '&partial=false&symbol=' + sym
        url += '&count=' + str(count) + '&start=' + str(start) + '&reverse=true'
        r = requests.get(url, allow_redirects=True)
        res=proc_data(r.content)
        #print res

        data += res
        iterations -= 1     
    return data


Feed.init()
Instrument.init()
Prediction.init()
Roi.init()
frequency=60

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


def readData(instrument_id, path_datasets, symbol, period=60, count=500, iterations=1):
    es = pyelasticsearch.ElasticSearch(port=int(es_port))
    
    '''
    def documents():
        feed_list=Feed.search().filter('term',instrument_id=int(instrument_id))
        start=0
        page=start + 1000
        end=feed_list.count()
        while start < end:
            if page > end:
                page = end
            feed_list=feed_list[start:page]
            for feed in feed_list:
                #feed.delete()
                
                yield es.delete_op(id=feed.meta.id,index=idx_name,doc_type='feed')
            start += 1000
            page = start + 1000
    

    for chunk in bulk_chunks(documents(),
         docs_per_chunk=500,
         bytes_per_chunk=10000):
        try:
            es.bulk(chunk, index=idx_name)
        except Exception as e:
            print (e)

    '''
    frequency=60
    last_feed_date = datetime.now() - relativedelta(years=10)
    try:
        last_feed_date=Feed.search().filter('term',instrument_id=int(instrument_id)).filter('term',frequency=frequency).sort('-date').execute()[0].date
        last_feed_date = datetime.now() - relativedelta(days=30)
    except Exception as e:
        print (e)
        try:
            if frequency <= 60:
                last_feed_date = datetime.now() - relativedelta(days=30)
        except Exception as e:
            print (e)

    
    print ('last feed date', last_feed_date)
    
    def documents2():
        date_now=''
        
        js=get_blockchain(symbol, period, count, iterations)
        for line in js:    
            if line['high'] and line['low'] and line['close'] and line['open']:
                i=0
                i+=1
                row=dict()
                row['timestamp']=line['timestamp']
                row['low']=float(line['low'])
                row['high']=float(line['high'])
                row['open']=float(line['open'])
                row['close']=float(line['close'])
                if line['vwap']:
                    row['wap']=float(line['vwap'])
                else:
                    row['wap']=float(line['close'])
                row['amount']=float(line['volume'])
                volume=row['amount']
                high_price=row['high']
                low_price=row['low']
                close_price=row['close']
                open_price=row['open']

                timestamp=row['timestamp']
                date=datetime.fromtimestamp(row['timestamp'])
                        
                if date > last_feed_date:
                    
                    if frequency == 1:
                        date_str=(date).strftime('%Y-%m-%dT%H:%M:%S')
                    if frequency == 60:
                        date_str=date.strftime('%Y-%m-%dT%H:%M:00')
                    elif frequency == 3600:
                        date_str=(date).strftime('%Y-%m-%dT%H:00:00')
                        
                    timestamp=time.mktime(parse(date_str).timetuple())
                    
                    print timestamp, date_str, row['timestamp']
                    if date_str != date_now:
                        if date_str and open_price and high_price and low_price and close_price and volume:
                            mydoc='main.feed.' + str(instrument_id) + '|' + str(timestamp) + '|' + str(frequency)
                            data={
                                    'instrument_id':int(instrument_id),
                                    'frequency':frequency,
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
                            yield es.update_op(doc=data,
                                       id=mydoc, 
                                       index=idx_name,
                                       doc_type='feed',
                                       doc_as_upsert=True)
                            
                        date_now=date_str
                            
        
    for chunk in bulk_chunks(documents2(),
     docs_per_chunk=500,
     bytes_per_chunk=10000):
        try:
            es.bulk(chunk, index=idx_name)
        except Exception as e:
            print (e)


def    getInitialData(sym, iterations):
        # 500 * iterations records
        path_datasets='./data/'
        #inst=getInstrument('BTC_USD', 'BTC_USD', 'Blockchain', 'Blockchain', 'BTC_USD')
        #btc_usd = readCSV(inst.id, path_datasets, 'BTC_USD')
        inst=getInstrument(sym, sym, 'Blockchain', 'Blockchain', sym)
        btc_usd = readData(inst.id, path_datasets, sym, 60, 500, iterations)
        return inst.id

def    getLatestData(sym, inst_id):
        path_datasets='./data/'
        btc_usd = readData(inst_id, path_datasets, sym, 60, 3, 1)
    
def    main():
        sym='XBTM18'
        inst_id=getInitialData(sym, 25)
        while 1:
            try:
                print ('Getting Latest Data History')
                getLatestData(sym, inst_id)
                time.sleep(15)
            except Exception as e:
                print (e)   
                
if    __name__    ==    "__main__":
    try:
        main()
    except    KeyboardInterrupt:
        pass





