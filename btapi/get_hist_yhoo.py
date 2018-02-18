from yahoo_finance import Share, Currency
from pprint import pprint

yahoo = Currency('BTCUSD')
pprint(yahoo.get_historical('2015-04-25', '2016-02-29'))
