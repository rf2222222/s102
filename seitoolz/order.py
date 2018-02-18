import numpy as np
import pandas as pd
import time
import    sys
import    os
sys.path.append("../")

import json
from pandas.io.json import json_normalize
from coinbase_api.get_exec import get_coinbase_pos
from coinbase_api.place_order import place_order as place_coinbase_order
from bitmex_api.get_exec import get_bitmex_pos, get_bitmex_pos_price
from bitmex_api.place_order import place_order as place_bitmex_order
from quoine_api.get_exec import get_quoine_pos
from quoine_api.place_order import place_order as place_quoine_order
from seitoolz.signal import get_model_pos
from time import gmtime, strftime, time, localtime, sleep
import logging

        
def adj_size_coinbase(model_pos, system, systemname, sym, qty):
    system_pos=model_pos.loc[system]
   
    logging.warning('==============')
    logging.warning('Strategy:' + systemname)
    logging.warning('system_pos:' +str(system_pos))
    logging.warning("  Signal Name: " + system)
    
    pos_qty=get_coinbase_pos(sym)
    system_pos_qty=round(system_pos['action']) * qty
    
    logging.warning( "system pos: " + str(system_pos_qty) )
    logging.warning( "actual pos: " + str(pos_qty) )
    if system_pos_qty > pos_qty:
        quant=float(system_pos_qty - pos_qty)
        logging.warning( 'BUY: ' + str(quant) )
        place_coinbase_order('BUY', sym, quant)
    if system_pos_qty < pos_qty:
        quant=float(pos_qty - system_pos_qty)
        logging.warning( 'SELL: ' + str(quant) )
        place_coinbase_order('SELL', sym, quant)


def adj_size_bitmex(model_pos, system, systemname, sym, qty):
    system_pos=model_pos.loc[system]
   
    logging.warning('==============')
    logging.warning('====BITMEX====')
    logging.warning('Strategy:' + systemname)
    logging.warning('system_pos:' +str(system_pos))
    logging.warning("  Signal Name: " + system)
    
    pos_qty=get_bitmex_pos(sym)
    system_pos_qty=round(system_pos['action']) * qty
    
    logging.warning( "system pos: " + str(system_pos_qty) )
    logging.warning( "actual pos: " + str(pos_qty) )
    if system_pos_qty > pos_qty:
        quant=float(system_pos_qty - pos_qty)
        logging.warning( 'BUY: ' + str(quant) )
        place_bitmex_order('BUY', sym, quant)
    if system_pos_qty < pos_qty:
        quant=float(pos_qty - system_pos_qty)
        logging.warning( 'SELL: ' + str(quant) )
        place_bitmex_order('SELL', sym, quant)
        
def adj_size_quoine(model_pos, system, systemname, sym, qty):
    system_pos=model_pos.loc[system]
   
    logging.warning('==============')
    logging.warning('====quoine====')
    logging.warning('Strategy:' + systemname)
    logging.warning('system_pos:' +str(system_pos))
    logging.warning("  Signal Name: " + system)
    
    pos_qty=get_quoine_pos(sym)
    system_pos_qty=round(system_pos['action']) * qty
    
    logging.warning( "system pos: " + str(system_pos_qty) )
    logging.warning( "actual pos: " + str(pos_qty) )
    if system_pos_qty > pos_qty:
        quant=float(system_pos_qty - pos_qty)
        logging.warning( 'BUY: ' + str(quant) )
        place_quoine_order('BUY', sym, quant)
    if system_pos_qty < pos_qty:
        quant=float(pos_qty - system_pos_qty)
        logging.warning( 'SELL: ' + str(quant) )
        place_quoine_order('SELL', sym, quant)
        
