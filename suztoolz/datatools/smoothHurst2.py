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
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from cycler import cycler
import seaborn as sns
from itertools import cycle
from suztoolz.transform import garch, roofingFilter, softmax,\
            softmax_score
from sklearn.preprocessing import scale, robust_scale, minmax_scale

def smoothHurst2(p,bars,threshold=0.5, showPlot=False):
    index = p.index
    returns = p.pct_change().fillna(0)

    #print rfGarch.shape, p.shape
    if type(p) is pd.core.series.Series:
        p = p.values
        
    nrows=p.shape[0]
    Dimen=np.zeros(nrows)
    Hurst=np.zeros(nrows)
    SmoothHurst=np.zeros(nrows)
    gar=np.zeros(nrows)
    gar[:bars]=garch(returns[:bars])
    SmoothGarch=np.zeros(nrows)
    minmaxGarch=np.zeros(nrows)
    if bars%2>0:
        bars=bars-1
        
    a1 = math.exp(-math.sqrt(2)*math.pi/10.0)
    b1 = 2.0*a1*math.cos(math.sqrt(2)*math.radians(180)/10.0)

    c3 = -a1*a1
    c2 = b1
    c1 = 1-c2-c3

    for i,lb in enumerate(range(bars, nrows)):
        #print i,lb
        N3 = (max(p[i:lb]) - min(p[i:lb]))/float(bars)
        N2 = (max(p[i:lb-bars/2]) - min(p[i:lb-bars/2]))/float(bars/2)
        #print i, lb-bars/2, p[i:lb-bars/2]
        N1 = (max(p[lb-bars/2:lb]) - min(p[lb-bars/2:lb]))/float(bars/2)
        #print p[lb-bars/2:lb]
        if N1>0 and N2>0 and N3>0:
            Dimen[lb] = .5*((log(N1+N2)-log(N3))/log(2)+Dimen[lb-1])
        Hurst[lb]=2-Dimen[lb]
        #print Hurst
        SmoothHurst[lb]=c1*(Hurst[lb]+Hurst[lb-1])/2+c2*SmoothHurst[lb-1]\
                                +c3*SmoothHurst[lb-2]
        gar[lb] = garch(returns[:lb])[-1]
        minmaxGarch[lb] =minmax_scale(gar[i:lb]) [-1]
        #print gar[lb], len(returns[:lb])
        
    #SmoothGarch= roofingFilter(gar,bars)
    #SmoothGarch = np.nan_to_num(SmoothGarch)
    #softmaxGarch = minmax_scale
    #softmaxGarch = softmax(gar,bars,1)
    #softmaxGarch = softmax_score(gar)
    #print minmaxGarch
    #print gar, SmoothGarch
                                
    #to return
    #SmoothHG = np.maximum(SmoothHurst,minmaxGarch)
    SmoothHG = np.minimum(SmoothHurst,minmaxGarch)
    modes = np.where(SmoothHG<threshold,0,1)
    #modes = np.where(minmaxGarch<threshold,0,1)
    #mode2 = np.where(Hurst[bars:]<threshold,0,1)
    if showPlot:
        mode = modes[bars:]
        p=p[bars:]
        nrows=p.shape[0]
        SmoothHurst=SmoothHurst[bars:]
        #SmoothGarch=SmoothGarch[bars:]
        SmoothHG=SmoothHG[bars:]
        minmaxGarch=minmaxGarch[bars:]
        gar=gar[bars:]
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, nrows - 1)
            return index[thisind].strftime("%Y-%m-%d %H:%M")
            

        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(0, len(p)), ylim=(p.min()*0.99, p.max()*1.01))
        ax.plot(np.arange(len(p)), p, 'r-', alpha=0.5)
        ax.scatter(np.arange(len(p))[mode == 0], p[mode == 0], color='g', label='0 CycleMode')
        ax.scatter(np.arange(len(p))[mode == 1], p[mode == 1], color='r', label='1 TrendMode')

        handles, labels = ax.get_legend_handles_labels()
        lgd2 = ax.legend(handles, labels, loc='upper right',prop={'size':10})
        #ax.plot(np.arange(len(p))[self.pivots != 0], p[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(p))[self.pivots == 1], p[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(p))[self.pivots == -1], p[self.pivots == -1], color='r')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        #annotate last index
        ax.annotate(index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
                    
        ax2=ax.twinx()
        #same color to dashed and non-dashed
        ax2.set_color_cycle(sorted(sns.color_palette("husl", 2)))
        #ax2.plot(np.arange(len(p)),SmoothHurst,color='b', label='smoothHurst')
        #ax2.plot(np.arange(len(p)),SmoothGarch,color='c', label='smoothGarch')
        ax2.plot(np.arange(len(p)),SmoothHG,color='y', label='smoothHG')
        #ax2.plot(np.arange(len(p)),minmaxGarch,color='c', label='minmaxGarch')
        #ax2.plot(np.arange(len(p)),gar,color='y', label='Garch')
        handles, labels = ax2.get_legend_handles_labels()
        lgd2 = ax2.legend(handles, labels, loc='lower right',prop={'size':10})
        #ax2.scatter(np.arange(len(p))[mode2 == 0], p[mode2 == 0], color='b', label='hurstCycleMode')
        #ax2.scatter(np.arange(len(p))[mode2 == 1], p[mode2 == 1], color='y', label='hurstTrendMode')

        #ax2.plot(np.arange(nrows),dpsEquity, label='dps '+system, ls=next(linecycle))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        if mode[-1] ==0:
            title = 'In Cycle Mode '
        else:
            title = 'In Trend Mode '
        ax2.set_title(title+str(bars)+' bars, threshold '+str(threshold))
        ax.set_xlim(0, nrows)        
        ax2.set_xlim(0, nrows)       
        fig.autofmt_xdate()        
        plt.show()
        
    return modes
    
if __name__ == "__main__":
    p=dataSet.Close
    bars=90
    threshold=.2
    smoothHurst2(p,bars,threshold=threshold, showPlot=True)