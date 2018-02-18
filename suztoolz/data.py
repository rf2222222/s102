import pandas as pd
import numpy as np
import sys
import datetime
from datetime import datetime as dt
from threading import Event
from pytz import timezone
from swigibpy import EWrapper, EPosixClientSocket, Contract
#!/usr/bin/python
# -*- coding: utf-8 -*-

# data.py

import datetime
import os, os.path
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

#from event import MarketEvent


class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVI) for each symbol requested. 

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the 
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, open, high, low, 
        close, volume, open interest).
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface. 
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True       
        self.bar_index = 0

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from DTN IQFeed. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                                      os.path.join(self.csv_dir, '%s.csv' % s),
                                      header=0, index_col=0, parse_dates=True,
                                      names=['datetime','open','low','high','close', 'volume','oi']
                                  ).sort()

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].\
                reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print "That symbol is not available in the historical data set."
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the 
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print "That symbol is not available in the historical data set."
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = self._get_new_bar(s).next()
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())

#saveSignals= False
WAIT_TIME = 60.0



class HistoricalDataExample(EWrapper):
    '''Callback object passed to TWS, these functions will be called directly
    by TWS.
    '''

    def __init__(self):
        super(HistoricalDataExample, self).__init__()
        self.got_history = Event()
        self.data = pd.DataFrame(columns = ['Open','High','Low','Close','Volume'])

    def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFilledPrice, clientId, whyHeld):
        pass

    def openOrder(self, orderID, contract, order, orderState):
        pass

    def nextValidId(self, orderId):
        '''Always called by TWS but not relevant for our example'''
        pass

    def openOrderEnd(self):
        '''Always called by TWS but not relevant for our example'''
        pass

    def managedAccounts(self, openOrderEnd):
        '''Called by TWS but not relevant for our example'''
        pass

    def getData(self):
        return self.data;
        
    def historicalData(self, reqId, date, open, high,
                       low, close, volume,
                       barCount, WAP, hasGaps):

        if date[:8] == 'finished':
            print("History request complete")
            self.got_history.set()
        else:
            self.data.loc[date] = [open,high,low,close,volume]
            #print "History %s - Open: %s, High: %s, Low: %s, Close: %s, Volume: %d"\
            #          % (date, open, high, low, close, volume)

            #print(("History %s - Open: %s, High: %s, Low: %s, Close: "
            #       "%s, Volume: %d, Change: %s, Net: %s") % (date, open, high, low, close, volume, chgpt, chg));

        #return self.data


def getDataFromIB(brokerData,endDateTime):
    #data_cons = pd.DataFrame()
    # Instantiate our callback object
    callback = HistoricalDataExample()

    # Instantiate a socket object, allowing us to call TWS directly. Pass our
    # callback object so TWS can respond.
    tws = EPosixClientSocket(callback)
    #tws = EPosixClientSocket(callback, reconnect_auto=True)
    # Connect to tws running on localhost
    if not tws.eConnect("", brokerData['port'], brokerData['client_id']):
        raise RuntimeError('Failed to connect to TWS')

    # Simple contract for GOOG
    contract = Contract()
    contract.exchange = brokerData['exchange']
    contract.symbol = brokerData['symbol']
    contract.secType = brokerData['secType']
    contract.currency = brokerData['currency']
    ticker = contract.symbol+contract.currency
    #today = dt.today()

    print("\nRequesting historical data for %s" % ticker)

    # Request some historical data.

    #for endDateTime in getHistLoop:
    tws.reqHistoricalData(
        brokerData['tickerId'],                                         # tickerId,
        contract,                                   # contract,
        endDateTime,                            #endDateTime
        brokerData['durationStr'],                                      # durationStr,
        brokerData['barSizeSetting'],                                    # barSizeSetting,
        brokerData['whatToShow'],                                   # whatToShow,
        brokerData['useRTH'],                                          # useRTH,
        brokerData['formatDate']                                          # formatDate
        )


    print("====================================================================")
    print(" %s History requested, waiting %ds for TWS responses" % (endDateTime, WAIT_TIME))
    print("====================================================================")


    try:
        callback.got_history.wait(timeout=WAIT_TIME)
    except KeyboardInterrupt:
        pass
    finally:
        if not callback.got_history.is_set():
            print('Failed to get history within %d seconds' % WAIT_TIME)
    
    #data_cons = pd.concat([data_cons,callback.data],axis=0)
             
    print("Disconnecting...")
    tws.eDisconnect()
        
    return callback.data
