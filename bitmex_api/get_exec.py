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


def get_bitmex_pos_price():
        
    # Basic authenticated call
    print('\n---A basic Position GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    res, http_response = bitMEXAuthenticated.Position.Position_get().result()
    #pp.pprint(res)
    price=0
    prices=[]
    for bal in res:
        try:
            if abs(float(bal['simpleQty'])) > 0.08:
                price={'symbol':bal['symbol'], 'price': bal['avgEntryPrice'], 'qty': float(bal['simpleQty'])}
                prices.append(price)
        except Exception as e:
            print (e)
    return prices

def get_bitmex_portfolio_qty(symbol):
        
    # Basic authenticated call
    print('\n---A basic Position GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    res, http_response = bitMEXAuthenticated.Position.Position_get(filter=json.dumps({'symbol': symbol})).result()
    #pp.pprint(res)
    qty=0
    for bal in res:
        try:
            if bal['symbol'] == symbol:
                qty += float(bal['simpleQty'])
        except Exception as e:
            print (e)
    return qty

    

def get_bitmex_open_order_qty(symbol):
    # Basic authenticated call
    print('\n---A basic open orders GET:---')
    print('The following call requires an API key. If one is not set, it will throw an Unauthorized error.')
    res, http_response = bitMEXAuthenticated.Order.Order_getOrders(filter=json.dumps({'symbol': symbol, 'open': True})).result()
    #pp.pprint(res)
    qty=0
    if len(res) > 0:
        
        for order in res:
            try:
                print order
                if order['side'] == 'Buy':
                    qty+=float(order['simpleOrderQty'])
                if order['side'] == 'Sell':
                    qty-=float(order['simpleOrderQty'])
            except Exception as e:
                print (e)
        #try:
        #    bitMEXAuthenticated2.Order.Order_cancelAll(symbol=symbol).result()
        #    time.sleep(2)
        #except Exception as e:
        #    print (e)
        #    time.sleep(2)
    return qty
        
def get_bitmex_pos(symbol):
    order_qty=get_bitmex_open_order_qty(symbol)
    portf_qty=get_bitmex_portfolio_qty(symbol)
    qty=portf_qty + order_qty
    return qty
