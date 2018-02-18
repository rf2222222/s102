# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""
import math
import numpy as np
import pandas as pd
import talib as ta
import arch
import random
import time
from os import listdir
from os.path import isfile, join
from suztoolz.datatools.zigzag2 import zigzag as zg
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import statsmodels.tsa.stattools as ts
from pandas.stats.api import ols
from matplotlib.dates import DateFormatter, WeekdayLocator,\
                                            MonthLocator, MONDAY, HourLocator, date2num
class zigzag(object):
    '''
    	all list parameters are expected to be an one dimensional
    	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
    '''
    def __init__(self, prices, up_thresh, down_thresh):
        self.prices = prices
        #self.candlesticks=zip(date2num(prices.index.to_pydatetime()),prices['Open'],prices['High'],prices['Low'],prices['Close'],prices['Volume'])
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.pivots = self.peak_valley_pivots()
        self.initial_pivot = self._identify_initial_pivot()
        self.eCurves = {}
      
    def peak_valley_pivots(self):
        """
        Finds the peaks and valleys of a series.
        """
        #print 'dt', self.down_thresh
        if self.down_thresh > 0:
            raise ValueError('The down_thresh must be negative.')
    
        initial_pivot = self._identify_initial_pivot()
    
        t_n = len(self.prices)
        pivots = np.zeros(t_n, dtype='i1')
        pivots[0] = initial_pivot
    
        # Adding one to the relative change thresholds saves operations. Instead
        # of computing relative change at each point as x_j / x_i - 1, it is
        # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
        # This saves (t_n - 1) subtractions.
        up_thresh = 1 +self.up_thresh
        down_thresh = 1 + self.down_thresh
    
        trend = -initial_pivot
        last_pivot_t = 0
        last_pivot_x = self.prices[0]
        for t in range(1, len(self.prices)):
            x = self.prices[t]
            r = x / last_pivot_x
    
            if trend == -1:
                if r >= up_thresh:
                    pivots[last_pivot_t] = trend
                    trend = 1
                    last_pivot_x = x
                    last_pivot_t = t
                elif x < last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
            else:
                if r <= down_thresh:
                    pivots[last_pivot_t] = trend
                    trend = -1
                    last_pivot_x = x
                    last_pivot_t = t
                elif x > last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
    
        if last_pivot_t == t_n-1:
            pivots[last_pivot_t] = trend
        elif pivots[t_n-1] == 0:
            pivots[t_n-1] = -trend
    
        return pivots
        
    def _identify_initial_pivot(self):
        """Quickly identify the X[0] as a peak or valley."""
        PEAK, VALLEY = 1, -1
        x_0 = self.prices[0]
        max_x = x_0
        max_t = 0
        min_x = x_0
        min_t = 0
        up_thresh = 1 +self.up_thresh
        down_thresh = 1 + self.down_thresh
    
        for t in range(1, len(self.prices)):
            x_t = self.prices[t]
    
            if x_t / min_x >= up_thresh:
                return VALLEY if min_t == 0 else PEAK
    
            if x_t / max_x <= down_thresh:
                return PEAK if max_t == 0 else VALLEY
    
            if x_t > max_x:
                max_x = x_t
                max_t = t
    
            if x_t < min_x:
                min_x = x_t
                min_t = t
    
        t_n = len(self.prices)-1
        return VALLEY if x_0 < self.prices[t_n] else PEAK
        
    def compute_segment_returns(self):
        """Return a numpy array of the pivot-to-pivot returns for each segment."""
        pivot_points = np.array([self.prices[i] for i,x in enumerate(~np.equal(self.pivots,0)) if x == True])
        return pivot_points[1:] / pivot_points[:-1] - 1.0
    
    def pivots_to_modes(self):
        """
        Translate pivots into trend modes.
        Parameters
        ----------
        pivots : the result of calling peak_valley_pivots
        Returns
        -------
        A numpy array of trend modes. That is, between (VALLEY, PEAK] it is 1 and
        between (PEAK, VALLEY] it is -1.
        """
        modes = np.zeros(len(self.pivots), dtype='i1')
        modes[0] = self.pivots[0]
        mode = -modes[0]
        for t in range(1, len(self.pivots)):
            x = self.pivots[t]
            if x != 0:
                modes[t] = mode
                mode = -x
            else:
                modes[t] = mode
        return modes
    
    def max_drawdown(self):
        """
        Return the absolute value of the maximum drawdown of sequence X.
        Note
        ----
        If the sequence is strictly increasing, 0 is returned.
        """
        mdd = 0
        peak = self.prices[0]
        for x in self.prices:
            if x > peak: 
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return mdd
        

def seasonalClassifier(ticker, dataPath, **kwargs):
    #plt.style.use('ggplot')
    l=kwargs.get('l',8)
    w=kwargs.get('w',8)
    lb=kwargs.get('lb',270)
    zzpstd=kwargs.get('zzpstd',2.5)
    zzsstd=kwargs.get('zzsstd',3.5)
    zs_window=kwargs.get('zs_window',60)
    rc_window=kwargs.get('rc_window',10)
    pivotDateLookforward=kwargs.get('pivotDateLookforward',3)
    minValidationLength=kwargs.get('minValidationLength',5)
    savePath = kwargs.get('savePath',None)
    #atrPath = kwargs.get('atrPath', './data/futuresATR.csv')
    debug = kwargs.get('debug',False)
    
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    auxFutures = [x.split('_')[0] for x in files]
    for symbol in auxFutures:    
        if 'YT' not in symbol:
            contract = ''.join([i for i in symbol.split('_')[0] if not i.isdigit()])
        else:
            contract=symbol.split('_')[0]
        

        if ticker==contract:
            #if 'F_'+contract+'.txt' in files and (ticker[0:3] in contract or ticker[3:6] in contract):
            filename = symbol+'_B.CSV'
            data = pd.read_csv(dataPath+filename, index_col=0, header=None)[-lb-zs_window:]
            
            #data = data.drop([' P',' R', ' RINFO'],axis=1)
            #data = ratioAdjust(data)
            #data.index = date2num(data.index.to_pydatetime())
            data.index = pd.to_datetime(data.index,format='%Y%m%d').to_pydatetime()
            data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            data.index.name = 'Dates'
            data2=data
            data=data[-lb:]
            #contract = ''.join([i for i in contract if not i.isdigit()])
            index=data.index
            
            #chart format
            def format_date(x, pos=None):
                thisind = np.clip(int(x + 0.5), 0, lb - 1)
                #print thisind,index
                return index[thisind].strftime("%Y-%m-%d %H:%M")
                
            def align_yaxis(ax1, v1, ax2, v2):
                """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
                _, y1 = ax1.transData.transform((0, v1))
                _, y2 = ax2.transData.transform((0, v2))
                inv = ax2.transData.inverted()
                _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
                miny, maxy = ax2.get_ylim()
                ax2.set_ylim(miny+dy, maxy+dy)
                
            major = MonthLocator()        # major ticks on the mondays
            minor = WeekdayLocator(MONDAY)              # minor ticks on the days
            majorFormat = DateFormatter('%b %Y')  # e.g., Jan 12
            #minorFormat = DateFormatter('%d')      # e.g., 12
            
           #data points
            pstd=data.Close.pct_change().std()
            zzp = zigzag(data.Close.values,pstd*zzpstd,pstd*-zzpstd)
            zzp_pivots=zzp.peak_valley_pivots()
            
            sstd=data.S.pct_change().std()
            zzs = zigzag(data.S.values,sstd*zzsstd,sstd*-zzsstd)
            zzs_pivots=zzs.peak_valley_pivots()
            
            runs=pd.DataFrame(np.where(zzp.pivots_to_modes()==zzs.pivots_to_modes(),1,-1),columns=['normal'])
            runs['block'] = (runs['normal'] != runs['normal'].shift(1)).astype(int).cumsum()
            runs['count'] = runs.groupby('block').transform(lambda x: range(1, len(x) + 1))
            #correlated =data.index[(runs['count']==1) & (runs['normal'] ==1)]
            correlated =data.index[runs['normal'] ==1]
            #anticorrelated = data.index[(runs['count']==1) & (runs['normal'] ==-1)]
            anticorrelated = data.index[runs['normal'] ==-1]
            currRun = runs['normal'].iloc[-1]*runs['count'].iloc[-1]
            
            #uses the seasonality rather than the price because the last zzpivot for the price may change in the future
            i=-2
            #validationStartDate=data.index[np.nonzero(zzs_pivots)[0]][i]
            validationStartDate=data.index[np.nonzero(zzp_pivots)[0]][i]
            validationLength=len(data.ix[validationStartDate:])
            #print 'sea',data.index[np.nonzero(zzs_pivots)[0]],len(data.ix[data.index[np.nonzero(zzs_pivots)[0]][i]:]), validationStartDate
            #print 'price',data.index[np.nonzero(zzp_pivots)[0]],len(data.ix[data.index[np.nonzero(zzp_pivots)[0]][i]:]), validationStartDate
            while  validationLength<minValidationLength:
                i-=1
                validationStartDate=data.index[np.nonzero(zzp_pivots)[0]][i]
                validationLength=len(data.ix[validationStartDate:])
                #print 'price',data.index[np.nonzero(zzp_pivots)[0]],len(data.ix[data.index[np.nonzero(zzp_pivots)[0]][i]:])
                #print 'sea',data.index[np.nonzero(zzs_pivots)[0]],len(data.ix[data.index[np.nonzero(zzs_pivots)[0]][i]:]), validationStartDate
                #print 'price',data.index[np.nonzero(zzp_pivots)[0]],validationLength, validationStartDate
                #validationStartDate=data.index[np.nonzero(zzs_pivots)[0]][i]
            i=-2
            validationStartDate2=data.index[np.nonzero(zzs_pivots)[0]][i]
            validationLength2=len(data.ix[validationStartDate2:])

            while  validationLength2<minValidationLength:
                i-=1
                validationStartDate2=data.index[np.nonzero(zzs_pivots)[0]][i]
                validationLength2=len(data.ix[validationStartDate2:])
            
            #print validationStartDate, validationStartDate2, validationLength, validationLength2
            #print validationStartDate> validationStartDate2, validationLength> validationLength2
            if validationLength> validationLength2:
                validationStartDate =validationStartDate2
                validationLength=validationLength2
            #find next seasonal pivot, +5 for to lookahead of weekend/larger lookforward bias
            i=1 
            print data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime()
            pivotDate=(data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().year+1)*10000+\
                            data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().month*100\
                            +data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().day
            currentDate=data.index[-1].to_datetime().year*10000+data.index[-1].to_datetime().month*100\
                                +data.index[-1].to_datetime().day+pivotDateLookforward
            print i,pivotDate,currentDate, not currentDate<pivotDate
            while not currentDate<pivotDate:
                i+=1
                pivotDate=(data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().year+1)*10000+\
                                data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().month*100\
                                +data.index[np.nonzero(zzs_pivots)[0][i]].to_datetime().day
                print i,pivotDate,currentDate
            nextSea=round(data.S.iloc[np.nonzero(zzs_pivots)[0][i]],2)
            
            currSea=round(data.S[-1],2)
            if nextSea>currSea:
                seaBias=1
            else:
                seaBias=-1
            #ax3
            corr = pd.rolling_corr(data.Close.pct_change().dropna().values,\
                                            data.S.pct_change().shift(-1).dropna().values,window=rc_window) 
            corr=pd.ewma(pd.Series(np.insert(corr,0,np.nan), index=data.index),com=0.5)
            #ax4 spread
            res = ols(y=data2.Close, x=data2.S)
            spread=data2.Close-res.beta.x*data2.S        
            zs_spread= ((spread - pd.rolling_mean(spread,zs_window))/pd.rolling_std(spread,zs_window)).ix[data.index]
            
            #top axis
            fig = plt.figure(figsize=(w,l*2))
            fig.subplots_adjust(bottom=0.2)
            ax=plt.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
            #ax = fig.add_subplot(211, autoscale_on=False)
            ax.xaxis.set_major_formatter(majorFormat)
            #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_minor_locator(minor)   
            ax.plot(data.index, data.Close, 'b:', alpha=0.5, label=str(pivotDate)+' Bias '+str(seaBias))
            ax.plot(data.index[zzp_pivots != 0], data.Close[zzp_pivots != 0], alpha=0.4, color='c',ls='-',\
                        label='SRUN'+str(currRun)+' ZZ'+str(zzpstd))
            #v start
            ax.annotate('', (data.index[-validationLength], data.Close.iloc[-validationLength]),
                             arrowprops=dict(facecolor='magenta', shrink=0.03), xytext=(-20,0), textcoords='offset points',
                             size='medium', alpha=0.6)
            #lb
            ax.annotate('', (data.index[-zs_window], data.Close.iloc[-zs_window]),
                             arrowprops=dict(facecolor='violet', shrink=0.03), xytext=(-20,0), textcoords='offset points',
                             size='medium', alpha=0.6)
            ax.annotate(data.index[-zs_window].strftime("%Y-%m-%d %H:%M /")+\
                            validationStartDate.strftime(" %Y-%m-%d %H:%M"),\
                            xy=(0.56, 0.025), ha='left', va='top', xycoords='axes fraction', fontsize=12)        
            ax.yaxis.set_label_position("left")
            ax.set_ylabel('Price', size=12)
            ax.set_title(ticker+' Price vs. Seasonality')
            ax.grid(which='major', linestyle='-', color='white') 
            plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
            #ax.scatter(correlated, data.Close.ix[correlated], color='k', label=str(int((float(len(correlated))/lb)*100))+'% Correlated')
            #ax.scatter(anticorrelated, data.Close.ix[anticorrelated], color='r', label=str(int((float(len(anticorrelated))/lb)*100))+'% Anti-Correlated')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower right',prop={'size':10}, bbox_to_anchor=(1, .975))
            
            #top axis 2
            ax2p=ax.twinx()
            ax2p.plot(data.index, data.S, 'g:', alpha=0.5, label=str(currSea)+' Seasonality')
            ax2p.plot(data.index[zzs_pivots != 0], data.S[zzs_pivots != 0], alpha=0.8, color='green',ls='-',\
                                label=str(nextSea)+' ZZ'+str(zzsstd)+' Seasonality')
            ax2p.axhline(nextSea, color='magenta', alpha=0.6)
            ax2p.axhline(currSea, color='violet', alpha=0.8)
            #ax2p.scatter(np.arange(lb)[zzs_pivots == 1], data.S[zzs_pivots == 1], color='g')
            #ax2p.scatter(np.arange(lb)[zzs_pivots == -1], data.S[zzs_pivots == -1], color='r')
            #ax2p.set_xlim(0, lb)
            mask = (runs['normal'] != runs['normal'].shift(-1))
            for i in runs[mask].index:
                if runs['normal'][i]<0:
                    xytext=(0,0)
                    color='r'
                else:
                    xytext=(0,0)
                    color='k'
                    
                ax2p.annotate(str(runs['count'][i]), (data.index[i], data.S[i]),
                             xytext=xytext, textcoords='offset points',color=color,
                             size='medium')
                ax.annotate(str(runs['count'][i]), (data.index[i], data.Close[i]),
                             xytext=xytext, textcoords='offset points',color=color,
                             size='medium')
            handles, labels = ax2p.get_legend_handles_labels()
            ax2p.legend(handles, labels, loc='lower left',prop={'size':10}, bbox_to_anchor=(0, .97))
            ax2p.yaxis.set_label_position("right")
            ax2p.set_ylabel('Seasonality', size=12)
            ax2p.grid(which='major', linestyle='--', color='white') 
            
            
            #bottom axis 1
            ax3 = plt.subplot2grid((2,1), (1,0), rowspan=1, colspan=1, sharex=ax)
            #ax3 = fig.add_subplot(212, autoscale_on=False)
            ax3.xaxis.set_major_formatter(majorFormat)
            #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax3.xaxis.set_major_locator(major)
            ax3.xaxis.set_minor_locator(minor)
            #ax3.plot(data.index, np.nan_to_num(corr), 'r:', alpha=0.5)
            ax3.plot(data.index, corr, 'k:', alpha=0.5, label=str(round(corr.iloc[-1],2))+' Correlation lb'+str(rc_window)+' lag1')
            ax3.scatter(correlated, corr.ix[correlated], color='g', alpha=0.6,\
                            label=str(int((float(len(correlated))/lb)*100))+'% Correlated')
            ax3.scatter(anticorrelated, corr.ix[anticorrelated], color='r', alpha=0.6,\
                            label=str(int((float(len(anticorrelated))/lb)*100))+'% Anti-Correlated')
            ax3.yaxis.set_label_position("left")
            ax3.set_ylabel('Correlation', size=12)
            ax3.set_title(ticker+' Spread & Correlation')
            ax3.set_ylim((-1,1))
            #ax3.axhline(0, color='white')
            handles, labels = ax3.get_legend_handles_labels()
            ax3.legend(handles, labels, loc='lower left',prop={'size':10}, bbox_to_anchor=(0, .94))

            #annotate last index
            ax3.annotate(data.Close.index[-1].strftime("%Y-%m-%d %H:%M"),\
                        xy=(0.78, 0.025), ha='left', va='top', xycoords='axes fraction', fontsize=12)        

            
            plt.setp(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
            
            #bottom axis 2
            ax4=ax3.twinx()
            #ax4.plot(data.index, zs_spread, 'k-', alpha=0.5, label='ZS Spread lb'+str(zs_window))
            ax4.fill_between(data.index, zs_spread, color='#0079a3', alpha=0.4,\
                                    label=str(round(zs_spread[-1],2))+' ZS Spread lb'+str(zs_window)) 
            ax4.yaxis.set_label_position("right")
            ax4.set_ylabel('Spread', size=12)
            ax4.set_ylim((np.floor(min(zs_spread.fillna(0))),np.ceil(max(zs_spread.fillna(0)))))
            #ax4.grid(which='major', linestyle='--', color='white') 
            align_yaxis(ax3, 0, ax4, 0)
            handles, labels = ax4.get_legend_handles_labels()
            ax4.legend(handles, labels, loc='lower right',prop={'size':10}, bbox_to_anchor=(1, .97))
            ax4.set_xlim(data.index[0],data.index[-1])
            
            #annotate runs
            for i in runs[mask].index:
                if not np.isnan(corr[i]):
                    if runs['normal'][i]<0:
                        xytext=(2,-2)
                        color='r'
                    else:
                        xytext=(2,-2)
                        color='k'
                    #print runs['count'][i], (data.index[i], corr[i])
                    ax3.annotate(str(runs['count'][i]), (data.index[i], corr[i]),\
                                 xytext=xytext, textcoords='offset points',color=color, size='medium')
            #save/show plots
            if debug:
                plt.show()
            if savePath != None:
                print 'Saving '+savePath+'.png'
                fig.savefig(savePath+'.png', bbox_inches='tight')
                
            plt.close()
    return seaBias, currRun, data.Close.index[-1], validationStartDate

if __name__ == "__main__":
    version = 'v4'
    liveFutures =  [
                         #'AC',
                         #'AD',
                         #'AEX',
                         #'BO',
                         #'BP',
                         #'C',
                         #'CC',
                         #'CD',
                         #'CGB',
                         #'CL',
                         #'CT',
                         #'CU',
                         #'DX',
                         #'EBL',
                         #'EBM',
                         #'EBS',
                         #'ED',
                         #'EMD',
                         'ES',
                         #'FCH',
                         #'FC',
                         #'FDX',
                         #'FEI',
                         #'FFI',
                         #'FLG',
                         #'FSS',
                         #'FV',
                         #'GC',
                         #'HCM',
                         #'HG',
                         #'HIC',
                         #'HO',
                         #'JY',
                         #'KC',
                         #'KW',
                         #'LB',
                         #'LCO',
                         #'LC',
                         #'LGO',
                         #'LH',
                         #'LRC',
                         #'LSU',
                         #'MEM',
                         #'MFX',
                         #'MP',
                         #'MW',
                         #'NE',
                         #'NG',
                         #'NIY',
                         #'NQ',
                         #'O',
                         #'OJ',
                         #'PA',
                         #'PL',
                         #'RB',
                         #'RR',
                         #'RS',
                         #'S',
                         #'SB',
                         #'SF',
                         #'SI',
                         #'SIN',
                         #'SJB',
                         #'SM',
                         #'SMI',
                         #'SSG',
                         #'STW',
                         #'SXE',
                         #'TF',
                         #'TU',
                         #'TY',
                         #'US',
                         #'VX',
                         #'W',
                         #'YA',
                         #'YB',
                         #'YM',
                         #'YT2',
                         #'YT3'
                         ]
    ticker=liveFutures[0]
    #dataPath =  'Z:/TSDP/data/from_IB/'
    #dataPath = 'D:/data/tickerData/'
    dataPath = 'D:/ML-TSDP/data/csidata/v4futures2/'
    #atrPath='D:/ML-TSDP/data/'
    #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    #chartSavePath = None
    chartSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/simCharts/'+version+'_'+ticker
    
    sea = seasonalClassifier(ticker, dataPath, savePath=chartSavePath+'_MODE2', debug=True)

