import    sys
import    os
sys.path.append("../")

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bitmex_auth import APIKeyAuthenticator
import json
import pprint
import time

#HOST = "https://www.bitmex.com/api/v1"
#SPEC_URI = "https://www.bitmex.com/api/explorer/swagger.json"

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


request_client2 = RequestsClient()
request_client2.authenticator = APIKeyAuthenticator(HOST, API_KEY2, API_SECRET2)

bitMEXAuthenticated2 = SwaggerClient.from_url(
  SPEC_URI,
  config=config,
  http_client=request_client2)


# Place Order
def buyBitmex(sym, qty):
    #res, http_response = bitMEXAuthenticated.Quote.Quote_get(symbol=sym, reverse=True).result()
    res2, http_response2 = bitMEXAuthenticated.OrderBook.OrderBook_getL2(symbol=sym, depth=2).result()
    #print(res)
    total=0
    count=0
    #for quote in res:
    #    if quote['askPrice']:
    #        price=quote['askPrice']
    #        total+=float(price)
    #        count+=1
    #        print 'ask: ',price
            
    for quote in res2:
        if quote['side'] == 'Sell':
            print 'orderBook: ',quote['price']
            
            total+=float(quote['price'])
            count+=1
        if quote['side'] == 'Buy':
            print 'orderBook: ',quote['price']
            
            total+=float(quote['price'])
            count+=1
            
    price=round(total/count)
    orderQty=round(price * qty)
    if orderQty > 0:
        print ('Price: ', price, 'qty: ', qty, 'orderQty:', orderQty)
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Buy', orderQty=orderQty, price=price).result()
        res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Buy', orderQty=orderQty).result()
        #print(res)
    return True

def sellBitmex(sym, qty):
    #res, http_response = bitMEXAuthenticated.Quote.Quote_get(symbol=sym, reverse=True).result()
    res2, http_response2 = bitMEXAuthenticated.OrderBook.OrderBook_getL2(symbol=sym, depth=2).result()
    #print(res)
    total=0
    count=0
    #for quote in res:
    #    if quote['bidPrice']:
    #        price=quote['bidPrice']
    #        total+=float(price)
    #        count+=1
    #        print 'bid: ',price
            
    for quote in res2:
        if quote['side'] == 'Buy':
            print 'orderBook: ',quote['price']
            
            total+=float(quote['price'])
            count+=1
        if quote['side'] == 'Sell':
            print 'orderBook: ',quote['price']
            
            total+=float(quote['price'])
            count+=1
            
    price=round(total/count)
    orderQty=round(price * qty)
    if orderQty > 0:
        print ('Price: ', price, 'qty: ', qty, 'orderQty:', orderQty)
        # Basic order placement
        # print(dir(bitMEXAuthenticated.Order))
        #res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Sell', orderQty=orderQty, price=price).result()
        res, http_response = bitMEXAuthenticated.Order.Order_new(symbol=sym, side='Sell', orderQty=orderQty).result()
        #print(res)
    return True


def place_order(action, sym, qty):
    
    if action == 'BUY':
        buyBitmex(sym, qty)
    if action == 'SELL':
        sellBitmex(sym, qty)
        


