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

mpl.rcParams["axes.formatter.useoffset"] = False
class zigzag(object):
    '''
    	all list parameters are expected to be an one dimensional
    	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
    '''
    def __init__(self, prices, up_thresh, down_thresh):
        self.prices = prices
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
        
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, self.prices.shape[0] - 1)
            return self.prices.index[thisind].strftime("%Y-%m-%d %H:%M")
            
        fig = plt.figure(figsize=(w,l))
        ax = fig.add_subplot(111, xlim=(0, len(self.prices)), ylim=(self.prices.min()*0.99, self.prices.max()*1.01))
        #ax.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        #ax.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        ax.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        ax.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        ax.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        ax.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        if mode is not None:
            ax.scatter(np.arange(len(self.prices))[mode.values == 0], self.prices[mode.values == 0], color='g', label='0 CycleMode')
            ax.scatter(np.arange(len(self.prices))[mode.values == 1], self.prices[mode.values == 1], color='r', label='1 TrendMode')
        #annotate last index
        ax.annotate(self.prices.index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
        
        fig.autofmt_xdate()
        if startPeak is not None and startValley is not None:
            ax.annotate('peak start', startPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('valley start', startValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if majorPeak is not None and majorValley is not None:
            ax.annotate('major peak', majorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('major valley', majorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if minorPeak is not None and minorValley is not None:
            ax.annotate('minor peak', minorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('minor valley', minorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if shortStart is not None:
            ax.annotate('short start', shortStart,
            xytext=(-70, 0), textcoords='offset points',
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )            
        if cycleList is not None:
            for l in cycleList:
                 ax.annotate(str(l[0]), l[1],
                 xytext=(5, -5), textcoords='offset points',
                 size='medium')
                 
        if indicators is not None:
            ax.plot(np.arange(len(self.prices)), self.prices, 'ko', alpha=0.5)
            plt.rc('lines', linewidth=1)
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm','c']) +
                                       cycler('linestyle', ['-', '--', ':', '-.','-'])))
            ax2=ax.twinx()
            ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            
            for i,ind in enumerate(indicators):
                ax2.plot(np.arange(len(indicators)),indicators[ind], label=ind)
            handles, labels = ax2.get_legend_handles_labels()
            lgd2 = ax2.legend(handles, labels, loc='best',prop={'size':10})
            ax2.set_xlim(0, len(indicators))
            ax2.set_title(chartTitle)
            
        if signals is not None:
            plt.rc('lines', linewidth=2)
            #sns.palplot(sns.hls_palette(len(signals.columns), l=.3, s=.8))
            #plt.style.use('ggplot')
            linecycle = cycle(['-', '--'])
            #plt.rc('axes', prop_cycle=(cycler('color',\
            #            [plt.cm.cool(i) for i in np.linspace(0, 1, len(signals.columns))])))
            nrows = len(self.prices)
            ax2=ax.twinx()
            #same color to dashed and non-dashed
            ax2.set_color_cycle(sorted(sns.color_palette("husl", len(signals))*2))
            ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax2.set_title(chartTitle)
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
                ax2.plot(np.arange(nrows),nodpsEquity, label=system+' noDpsComm: '\
                                +str(nodpsComm)+' Sig: '+str(signal)+' Safef: '+str(nodpsSafef),\
                                ls=next(linecycle))
                ax2.plot(np.arange(nrows),dpsEquity, label='dps '+system+ ' dpsComm: '\
                            +str(dpsComm)+' Sig: '+str(signal)+' Safef: '+str(dpsSafef),\
                            ls=next(linecycle))

                    
            handles, labels = ax2.get_legend_handles_labels()
            lgd2 = ax2.legend(handles, labels, loc='lower left',prop={'size':10})
            ax2.set_xlim(0, nrows)
            #ax2.get_xaxis().get_major_formatter().set_useOffset(False)
            #ax2.get_xaxis().get_major_formatter().set_scientific(False)
                
        ax.set_xlim(0, len(self.prices))
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
