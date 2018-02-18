import psycopg2
import logging
from datetime import datetime
import urllib2
from xml.dom import minidom
import requests
from time import gmtime, strftime, localtime, sleep
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import time as mytime
import logging
import os
from coin import *
import sys

import json
import codecs
import StringIO
import re
from django.template.defaultfilters import slugify
from django.db import models
from django.core.paginator import Page
import os
from BeautifulSoup import BeautifulSoup
import os
from cookielib import LWPCookieJar
import requests
import cookielib
from pprint import pprint
from HTMLParser import HTMLParser
import elasticsearch
import pyelasticsearch
from pyelasticsearch import bulk_chunks

es = pyelasticsearch.ElasticSearch(port=9201)
from main.elasticmodels import *
from datetime import datetime
from dateutil.parser import parse
from forex_python.converter import CurrencyRates
#bsettings.configure(default_settings=beCOMPANY, DEBUG=True)

logging.basicConfig(filename='logs/update_currency.log',level=logging.DEBUG)

def removeTags(html):
    if html is not None:
	strip = lambda s: "".join(i for i in s if 1 < ord(i) < 130)
        h = HTMLParser()
        html=h.unescape(html)
        html=strip(html)

        html= re.sub(r'<.*?>', '\n', html)
        html= re.sub(r'<.*', '', html)
        html= re.sub(r'.*>', '', html)
        html= re.sub(r'&nbsp;', ' ', html)
        html= re.sub(r'&#39;', '', html)
        return html
    else:
        return ''  

def cleanseList(my_list):
    for list in my_list:
        list=re.sub('^[\s]*','',list)
        list=re.sub('[\s]*$','',list)
    return my_list

def stripped(html):
    if html is not None:
        if len(html) > 0:
            strip = lambda s: "".join(i for i in s if 1 < ord(i) < 130)
            h = HTMLParser()
            html=h.unescape(html)
            html=strip(html)
            return html
    return ''

def get_conversions(): 
    currencies=['KRW','JPY','MYR','SGD','HKD','IDR','THB','INR','CNY','PHP','EUR','GBP','RMB']
    from_currency='USD'
    conv=CurrencyRates()
    for currency in currencies:
        print 'From:',from_currency,' To:',currency
        cur=currency
        if cur=='RMB':
            cur='CNY'

        conversion=conv.get_rate(cur,from_currency)
        conversion2=conv.get_rate(from_currency,cur)
        
        if float(conversion2) == 0:
            conversion2 = 1/float(conversion)
        if float(conversion) == 0:
            conversion=1/float(conversion2)

        print "from_currency_price: 1",from_currency,"=",conversion,currency
        print "to_currency_price: $1=",conversion2,currency
            
        currency_list=CurrencyConversion.search().filter('term',**{'from_currency_name.raw':'USD'}).filter('term', **{'to_currency_name.raw':currency.upper()}).execute()
        if currency_list and len(currency_list) > 0:
            c=currency_list[0]
            print "Found",c.to_dict()
        else:
            c=CurrencyConversion()
        c.from_currency_name='USD'
        c.from_currency_price=float(conversion)
        c.to_currency_name=currency.upper()
        c.to_currency_price=float(conversion2)
        
        c.save()
        
    for currency in currencies:
        print 'From:',currency,' To:',from_currency
        cur=currency
        if cur=='RMB':
            cur='CNY'
        conversion=conv.get_rate(from_currency,cur)
        print "currency_price: 1",from_currency,"=",conversion,currency
        currency_list=Currency.search().filter('term',**{'currency_name.raw':currency.upper()}).execute()
        if currency_list and len(currency_list) > 0:
            c=currency_list[0]
            print "Found",c.to_dict()
            
        else:
            c=Currency()
        c.currency_name=currency.upper()
        c.currency_price=float(conversion)
        c.save()
        
get_conversions()