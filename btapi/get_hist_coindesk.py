import requests
import urllib
import urllib2

def get_hist_coindesk():
    #url = 'https://blockchain.info/charts/market-price?timespan=60days&format=json'
    #values = {'timespan' : '60days',
    #          'format' : 'json'}
    url = 'https://api.coindesk.com/v1/bpi/historical/close.json'
    values = {'index' : 'USD/CNY',
              'currency' : 'USD',
              'start' : '2014-01-01' #,'end' : '2016-03-05'
               }
    
    response = requests.get(url, params=values, json=values);
    #print response.text;
    return response.text;
