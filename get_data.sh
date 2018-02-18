#!/bin/sh
a=`pwd`
cd ~/coin/
rm *.gz
wget http://api.bitcoincharts.com/v1/csv/coinbaseUSD.csv.gz
gunzip coinbaseUSD.csv.gz
mv coinbaseUSD.csv BTC_USD_all.csv
tail -n 2000000 BTC_USD_all.csv > BTC_USD.csv
wget http://api.bitcoincharts.com/v1/csv/korbitKRW.csv.gz
gunzip korbitKRW.csv.gz
mv korbitKRW.csv BTC_KRW_all.csv
tail -n 2000000 BTC_KRW_all.csv > BTC_KRW.csv
cd $a
echo $a
python get_data_import.py

