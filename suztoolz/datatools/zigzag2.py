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
import statsmodels.tsa.stattools as ts
import datetime
from datetime import datetime as dt
from pytz import timezone
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy import stats
from sklearn.externals import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from cycler import cycler
import seaborn as sns
from itertools import cycle
from matplotlib.finance import candlestick_ohlc
from matplotlib.finance import volume_overlay3
from matplotlib.dates import num2date
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator,\
                                            DayLocator, MONDAY, HourLocator
import matplotlib.mlab as mlab

mpl.rcParams["axes.formatter.useoffset"] = False
class zigzag(object):
    '''
    	all list parameters are expected to be an one dimensional
    	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
    '''
    def __init__(self, prices, up_thresh, down_thresh):
        self.prices = prices.Close
        self.candlesticks=zip(date2num(prices.index.to_pydatetime()),prices['Open'],prices['High'],prices['Low'],prices['Close'],prices['Volume'])
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
        
    def plot_pivots(self, **kwargs):
        l=kwargs.get('l',6)
        w=kwargs.get('w',6)
        startPeak=kwargs.get('startPeak',None)
        startValley=kwargs.get('startValley',None)
        majorPeak=kwargs.get('majorPeak',None)
        majorValley=kwargs.get('majorValley',None)
        minorPeak=kwargs.get('minorPeak',None)
        minorValley=kwargs.get('minorValley',None)
        shortStart=kwargs.get('shortStart',None)
        cycleList = kwargs.get('cycleList',None)
        indicators = kwargs.get('indicators',None)
        signals = kwargs.get('signals',None)
        chartTitle = kwargs.get('chartTitle',1)
        mode = kwargs.get('mode',None)
        savePath = kwargs.get('savePath',None)
        debug = kwargs.get('debug',True)
        barsize=kwargs.get('barsize',None)
        
        fig = plt.figure(figsize=(w,l*2))
        #ax = fig.add_subplot(111)
        ax=plt.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
        plt.title(chartTitle)
        
        if barsize==None:
            width=0.6
            major = WeekdayLocator(MONDAY)        # major ticks on the mondays
            minor = DayLocator()              # minor ticks on the days
            majorFormat = DateFormatter('%b %d %Y')  # e.g., Jan 12
            minorFormat = DateFormatter('%d')      # e.g., 12
        else:
            #major = WeekdayLocator(MONDAY)        # major ticks on the mondays
            width=0.1
            minor = HourLocator(byhour=range(4,24,4))
            major = DayLocator()              # minor ticks on the days
            majorFormat = DateFormatter('%b %d %Y')  # e.g., Jan 12
            if len(self.prices)<30:
                minorFormat = DateFormatter('%H:%M')      
                ax.xaxis.set_minor_formatter(minorFormat)
                
            

        #plt.ylabel(ticker)
        #fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_major_formatter(majorFormat)
        dates = [x[0] for x in self.candlesticks]
        dates = np.asarray(dates)
        
        #volume
        volume = [x[5] for x in self.candlesticks]
        volume = np.asarray(volume)
        ax.fill_between(dates,0, volume, facecolor='#0079a3', alpha=0.4)
        #scale the x-axis tight
        #ax.set_xlim(min(dates),max(dates))
        # the y-ticks for the bar were too dense, keep only every third one
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::2])
        ax.yaxis.set_label_position("left")
        ax.set_ylabel('Volume', size=12)
        ax.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, horizontalalignment='right')
        #price candles
        ax2p=ax.twinx()
        candlestick_ohlc(ax2p, self.candlesticks, width=width, colorup='g')
        
        ax2p.xaxis_date()
        ax2p.autoscale_view()

        sma=pd.rolling_mean(self.prices,5)
        bbu=sma+pd.rolling_std(self.prices,5)
        bbl=sma-pd.rolling_std(self.prices,5)
        runs=pd.DataFrame(np.where(self.prices<sma,-1,1),columns=['col'])
        runs['block'] = (runs['col'] != runs['col'].shift(1)).astype(int).cumsum()
        runs['count'] = runs.groupby('block').transform(lambda x: range(1, len(x) + 1))
        
        #plt.rc('lines', linewidth=1)
        ax2p.plot(dates,sma,'k:',label='SMA5', linewidth=1.5)
        ax2p.plot(dates,bbu,'k-.',label='BBU', linewidth=1.5)
        ax2p.plot(dates,bbl,'k-.',label='BBL', linewidth=1.5)
        handles, labels = ax2p.get_legend_handles_labels()
        lgd = ax2p.legend(handles, labels, loc='best',prop={'size':10})
        ax2p.yaxis.set_label_position("right")
        ax2p.set_ylabel('Price', size=12)
        #print runs['count']
        for i,count in enumerate(runs['count']):
            if runs['col'][i]<0:
                xytext=(-3,-20)
            else:
                xytext=(-3,20)
                
            ax2p.annotate(str(count), (dates[i], self.prices[i]),
                         xytext=xytext, textcoords='offset points',
                         size='medium')
                
        #indicators
        ax3 = plt.subplot2grid((2,1), (1,0), rowspan=1, colspan=1)

        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, self.prices.shape[0] - 1)
            return self.prices.index[thisind].strftime("%Y-%m-%d %H:%M")

        #fig = plt.figure(figsize=(w,l))
        #ax3 = fig.add_subplot(111, xlim=(0, len(self.prices)), ylim=(self.prices.min()*0.99, self.prices.max3()*1.01))
        
        ax3.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        ax3.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        ax3.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        ax3.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')

        
        
        #ax3.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        #ax3.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        ax3.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        if mode is not None:
            ax3.scatter(np.arange(len(self.prices))[mode.values == 0], self.prices[mode.values == 0], color='g', label='0 CycleMode')
            ax3.scatter(np.arange(len(self.prices))[mode.values == 1], self.prices[mode.values == 1], color='r', label='1 TrendMode')
        #annotate last index
        ax3.annotate(self.prices.index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
        
        #fig.autofmt_xdate()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        if startPeak is not None and startValley is not None:
            ax3.annotate('peak start', startPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('valley start', startValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if majorPeak is not None and majorValley is not None:
            ax3.annotate('major peak', majorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('major valley', majorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if minorPeak is not None and minorValley is not None:
            ax3.annotate('minor peak', minorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('minor valley', minorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if shortStart is not None:
            ax3.annotate('short start', shortStart,
            xytext=(-70, 0), textcoords='offset points',
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )            
        if cycleList is not None:
            for l in cycleList:
                 ax3.annotate(str(l[0]), l[1],
                 xytext=(5, -5), textcoords='offset points',
                 size='medium')
                 
        if indicators is not None:
            #ax3.plot(np.arange(len(self.prices)), self.prices, 'ko', alpha=0.5)
            plt.rc('lines', linewidth=1)
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'm','y','c']) +\
                                       cycler('linestyle', ['-', '-', '-', '-.','--',':'])))
            ax4=ax3.twinx()
            ax4.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            
            for i,ind in enumerate(indicators):
                ax4.plot(np.arange(len(indicators)),indicators[ind], label=ind)
            handles, labels = ax4.get_legend_handles_labels()
            lgd2 = ax4.legend(handles, labels, loc='best',prop={'size':10})
            ax4.set_xlim(0, len(indicators))
            ax4.set_title(chartTitle)
            
        if signals is not None:
            plt.rc('lines', linewidth=2)
            #sns.palplot(sns.hls_palette(len(signals.columns), l=.3, s=.8))
            #plt.style.use('ggplot')
            linecycle = cycle(['-', '--'])
            #plt.rc('axes', prop_cycle=(cycler('color',\
            #            [plt.cm.cool(i) for i in np.linspace(0, 1, len(signals.columns))])))
            nrows = len(self.prices)
            ax4=ax3.twinx()
            #same color to dashed and non-dashed
            ax4.set_color_cycle(sorted(sns.color_palette("husl", len(signals))*2))
            ax4.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax4.set_title(chartTitle)
            #ga_pct = self.prices.pct_change().shift(-1).fillna(0)
            #ga_pct.name = 'gainAhead'
            
            for system in signals:
                nodpsEquity = signals[system][-nrows:].netEquity
                dpsEquity = signals[system][-nrows:].dpsNetEquity
                nodpsComm=round(signals[system][-nrows:].nodpsComm.sum(),0)
                dpsComm=round(signals[system][-nrows:].dpsCommission.sum(),0)
                signal=signals[system].signals[-1]
                nodpsSafef=signals[system].nodpsSafef[-1]
                dpsSafef=signals[system].dpsSafef[-1]
                #print system, chartTitle, nodpsEquity, dpsEquity
                ax4.plot(np.arange(nrows),nodpsEquity, label=system+' noDpsComm: '\
                                +str(nodpsComm)+' Sig: '+str(signal)+' Safef: '+str(nodpsSafef),\
                                ls=next(linecycle))
                ax4.plot(np.arange(nrows),dpsEquity, label='dps '+system+ ' dpsComm: '\
                            +str(dpsComm)+' Sig: '+str(signal)+' Safef: '+str(dpsSafef),\
                            ls=next(linecycle))

                    
            handles, labels = ax4.get_legend_handles_labels()
            lgd2 = ax4.legend(handles, labels, loc='lower left',prop={'size':10})
            ax4.set_xlim(0, nrows)
            #ax4.get_xaxis().get_major_formatter().set_useOffset(False)
            #ax4.get_xaxis().get_major_formatter().set_scientific(False)
                
        ax3.set_xlim(0, len(self.prices))
        
        if debug:
            plt.show()
        if savePath != None:
            print 'Saving '+savePath+'.png'
            fig.savefig(savePath+'.png', bbox_inches='tight')
            
        plt.close()
    
    def plot_hist(self):
        hist, bins = np.histogram(compute_segment_returns(self.prices,self.pivots), bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
