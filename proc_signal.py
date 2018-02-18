import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import json
from pandas.io.json import json_normalize
from seitoolz.signal import get_dps_model_pos, get_model_pos
from seitoolz.order import adj_size_coinbase, adj_size_bitmex, adj_size_quoine, get_bitmex_pos_price
from time import gmtime, strftime, localtime, sleep
import logging
from datetime import datetime

logging.basicConfig(filename='logs/proc_signal.log',level=logging.WARNING)



#subprocess.call(['python', 'get_ibpos.py'])
#ib_pos=get_ibpos()
#ib_pos=get_ibpos_from_csv()
    
def send_order():
    try:
        #currencyList=dict()
        #v1sList=dict()
        dpsList=dict()
        systemdata=pd.read_csv('./data/systems/system.csv')
        systemdata=systemdata.reset_index()
        for i in systemdata.index:
          system=systemdata.ix[i]
          dpsList[system['System']]=1
        
        #model_pos=get_model_pos(v1sList.keys())
        dps_model_pos=get_dps_model_pos(dpsList.keys())
        
        for i in systemdata.index:
            try:
                system=systemdata.ix[i]
                model=dps_model_pos
                try:
                    if system['quoine_submit']:
                        print ('Quoine:', model, system['System'], system['Name'], system['quoine_sym'], float(system['quoine_qty']))
                        adj_size_quoine(model, system['System'], system['Name'], system['quoine_sym'], float(system['quoine_qty']))
                        logging.warning(datetime.now())
                except Exception as f:
                    print (f)
                try:
                    if system['coinbase_submit']:
                        print ('Coinbase:', model, system['System'], system['Name'], system['coinbase_sym'], float(system['coinbase_qty']))
                        adj_size_coinbase(model, system['System'], system['Name'], system['coinbase_sym'], float(system['coinbase_qty']))
                        logging.warning(datetime.now())
                except Exception as f:
                    print (f)
                try:
                    if system['bitmex_submit']:
                        print ('bitmex:', model, system['System'], system['Name'], system['bitmex_sym'], float(system['bitmex_qty']))
                        adj_size_bitmex(model, system['System'], system['Name'], system['bitmex_sym'], float(system['bitmex_qty']))
                        logging.warning(datetime.now())
                except Exception as f:
                    print (f)
                    
            except Exception as e:
                logging.error("something bad happened", exc_info=True)
    except Exception as g:
        print (g)


def main():
    while 1:
        try:
            send_order();
            time.sleep(10)
        except Exception as e:
            print (e)
            
    
if    __name__    ==    "__main__":
    try:
        main()
    except    KeyboardInterrupt:
        pass


