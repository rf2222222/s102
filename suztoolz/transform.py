# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""
import math
import numpy as np
import arch
import pandas as pd
import talib as ta
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

def saveParams(start_time, dataSet, bestModelParams, bestParamsPath, modelSavePath,ticker, version, barSizeSetting,BSMdict):
    timenow, lastBartime, cycleTime = getCycleTime(start_time, dataSet)
    bestModelParams = bestModelParams.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
    bestModelParams = bestModelParams.append(pd.Series(data=lastBartime.strftime("%Y%m%d %H:%M:%S %Z"), index=['lastBartime']))
    bestModelParams = bestModelParams.append(pd.Series(data=cycleTime, index=['cycleTime']))
    
    print '\n'+'Saving Params for '+version
    files = [ f for f in listdir(bestParamsPath) if isfile(join(bestParamsPath,f)) ]
    if version+'_'+ticker+'_'+barSizeSetting+ '.csv' not in files:
        BMdf = pd.concat([bestModelParams,bestModelParams],axis=1).transpose()
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)
    else:
        BMdf = pd.read_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv').append(bestModelParams, ignore_index=True)
        #BMdf.index = BMdf.timestamp
        BMdf.to_csv(bestParamsPath+version+'_'+ticker+'_'+barSizeSetting+'.csv', index=False)
    
    print 'Saving Model for', version, 'validation period', bestModelParams.validationPeriod
    runName=BSMdict[bestModelParams.validationPeriod][0]
    print runName
    joblib.dump(BSMdict[bestModelParams.validationPeriod][1], modelSavePath+version+'_'+runName+'.joblib',compress=3)

def getCycleTime(start_time, dataSet, timeZone='US/Eastern'):
    timenow = dt.now(timezone(timeZone))
    lastBartime = timezone(timeZone).localize(dataSet.index[-1].to_datetime())
    #adjust cycletime if weekend
    weekday = dt.now(timezone(timeZone)).weekday()
    if weekday == 5 or weekday ==6:
        cycleTime = round(((time.time() - start_time)/60),2)
    else:
        cycleTime = (timenow-lastBartime).total_seconds()/60
        
    return timenow, lastBartime, cycleTime
    
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
        
    def plot_pivots(self, l=6, w=6,**kwargs):
        mp=kwargs.get('mp',None)
        mv=kwargs.get('mv',None)
        cycleList = kwargs.get('cycleList',None)
        indicators = kwargs.get('indicators',None)
        signals = kwargs.get('signals',None)
        initialEquity = kwargs.get('initialEquity',1)
        chartTitle = kwargs.get('chartTitle',1)
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
        fig.autofmt_xdate()
        if mp is not None and mv is not None:
            ax.annotate('major peak', mp,
            xytext=(-20, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('major valley', mv,
            xytext=(20, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if cycleList is not None and signals is None:
            for l in cycleList:
                 ax.annotate(str(l[0]), l[1],
                 xytext=(5, 0), textcoords='offset points',
                 size='medium')
                 
        if indicators is not None:
            plt.rc('lines', linewidth=1)
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                       cycler('linestyle', ['-', '--', ':', '-.'])))
            ax2=ax.twinx()
            ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            
            for i,ind in enumerate(indicators):
                ax2.plot(np.arange(len(indicators)),indicators[ind], label=ind)
            handles, labels = ax2.get_legend_handles_labels()
            lgd2 = ax2.legend(handles, labels, loc='best',prop={'size':10})
            ax2.set_xlim(0, len(indicators))
            
        if signals is not None:
            plt.rc('lines', linewidth=2)
            #sns.palplot(sns.hls_palette(len(signals.columns), l=.3, s=.8))
            #plt.style.use('ggplot')
            linecycle = cycle(['-', '--'])
            #plt.rc('axes', prop_cycle=(cycler('color',\
            #            [plt.cm.cool(i) for i in np.linspace(0, 1, len(signals.columns))])))
            nrows = len(self.prices)
            ax2=ax.twinx()
            ax2.set_color_cycle(sns.color_palette("husl", len(signals.columns)))
            ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax2.set_title(chartTitle)
            ga_pct = self.prices.pct_change().shift(-1).fillna(0)
            ga_pct.name = 'gainAhead'
            
            for col in signals:
                sig = pd.Series(data=0, index=self.prices.index)
                for idx in signals[col].index:
                   #print x,idx,signals[col][idx]
                   sig.set_value(idx,signals[col][idx])
                #  Compute cumulative equity for days with beShort signals    
                equityBeLongAndShortSignals = np.zeros(nrows)
                equityBeLongAndShortSignals[0] = initialEquity
                for i in range(1,nrows):
                    if (sig.iloc[i-1] < 0):
                        equityBeLongAndShortSignals[i] = (1+-ga_pct.iloc[i-1])*equityBeLongAndShortSignals[i-1]
                    elif (sig.iloc[i-1] > 0):
                        equityBeLongAndShortSignals[i] = (1+ga_pct.iloc[i-1])*equityBeLongAndShortSignals[i-1]
                    else:
                        equityBeLongAndShortSignals[i] = equityBeLongAndShortSignals[i-1]
                        
                self.eCurves[col]=equityBeLongAndShortSignals
            
            #add to axis
            for col in self.eCurves:
                ax2.plot(np.arange(nrows),equityBeLongAndShortSignals, label=col, ls=next(linecycle))
                    
            handles, labels = ax2.get_legend_handles_labels()
            lgd2 = ax2.legend(handles, labels, loc='best',prop={'size':10})
            ax2.set_xlim(0, nrows)                
                
        ax.set_xlim(0, len(self.prices))
        plt.show()
        plt.close()
    
    def plot_hist(self):
        hist, bins = np.histogram(compute_segment_returns(self.prices,self.pivots), bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

class ratio(object):
	'''
	all list parameters are expected to be an one dimensional
	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
	'''
	def __init__(self, prices, benchmark = None):
		self.prices = prices
		self.n = len(self.prices)
		self.benchmark = self._prepare_benchmark(benchmark)
		self.ret = np.diff(self.prices)
		self.b_ret = np.diff(self.benchmark)

	def sharpe(self):

		adj_ret = [a - b for a, b in zip(self.ret, self.b_ret)]
		std = np.std(self.ret)

		return self._get_info_ratio(adj_ret, std)

	def sortino(self):
		'''
		sortino is an adjusted ratio which only takes the 
		standard deviation of negative returns into account
		'''
		adj_ret = [a - b for a, b in zip(self.ret, self.b_ret)]
		avg_ret = np.mean(adj_ret)

		# Take all negative returns.
		neg_ret = [a ** 2 for a in adj_ret if a < 0]
		# Sum it.
		neg_ret_sum = np.sum(neg_ret)
		# And calculate downside risk as second order lower partial moment.
		down_risk = np.sqrt(neg_ret_sum / self.n)

		if down_risk > 0.0001:
			sortino = avg_ret / down_risk
		else:
			sortino = 0

		return sortino

	def _get_info_ratio(self, ret, std):

		avg = np.mean(ret)

		if std > 0.0001:
			return avg * np.sqrt(self.n) / std
		else:
			return 0

	def _prepare_benchmark(self, benchmark):

		if benchmark == None:
			benchmark = np.zeros(self.n)
		if len(benchmark) != self.n:
			raise Exception("benchmark mismatch")

		return benchmark
        
def roofingFilter(p,lb,bars=10.0):
    if lb <3:
        print 'lookback < 3. adjusting lookback minimum to 3'
        lb =3    
    if type(p) is pd.core.series.Series:
        p = p.values
        
    rad360 = math.radians(360)
    sqrt2div2 = math.sqrt(2)/2
    alpha = (math.cos(sqrt2div2*rad360/bars)+\
            math.sin(sqrt2div2*rad360/bars)-1)/\
            math.cos(rad360*sqrt2div2/bars)
    a1 = math.exp(-math.sqrt(2)*math.pi/10.0)
    b1 = 2.0*a1*math.cos(math.sqrt(2)*math.radians(180)/10.0)
    
    c3 = -a1*a1
    c2 = b1
    c1 = 1-c2-c3
    
    nrows = p.shape[0]
    highpass = np.zeros(nrows)
    
    for i in range(2,nrows):
        highpass[i] = (1-alpha/2.0)*(1-alpha/2.0)*\
                (p[i]-2*p[i-1]+p[i-2])+2*(1-alpha)*\
                highpass[i-1]-(1-alpha)*(1-alpha)*highpass[i-2]
    
    filt=np.zeros(nrows)
    for i in range(2,nrows):
        filt[i] = c1*(highpass[i]+highpass[i-1])/2.0+\
                c2*filt[i-1]+c3*filt[i-2]
    
    high=np.zeros(nrows)
    low=np.zeros(nrows)
    #soften ramp up
    for i in range(3,lb):
        high[i] = max(filt[2:i])
        low[i] = min(filt[2:i])
                
    stoch=np.zeros(nrows)
    rstoch=np.zeros(nrows)
    for i in range(lb,nrows):
        high[i] = max(filt[i-lb:i])
        low[i] = min(filt[i-lb:i])
        stoch[i] = (filt[i]-low[i])/(high[i]-low[i])
        rstoch[i] = c1*(stoch[i]+stoch[i-1])/2.0+\
                c2*rstoch[i-1]+c3*rstoch[i-2]
    return rstoch
    
def perturb_data(p,mean):
    #add guassian noise of rolling std lb
    if type(p) is pd.core.series.Series:
        p = p.values

    #rstd = pd.rolling_std(p,lb)
    #pc = ROC(p,lb)
    nrows = p.shape[0]
    purturbed_data = np.zeros(nrows)
    for i in range(1, nrows):
        perturb = p[i-1]*mean
        if perturb == 0:
            purturbed_data[i] = 0
        else:
            purturbed_data[i] = np.random.normal(0,perturb,1)
    #purturbed_data = np.ceil(purturbed_data)
    return p+purturbed_data
    
def garch(returns, verbose=False):
    #this one is fast but has a future leak.
    am = arch.arch_model(returns*100)
    #res = am.fit(iter=10)
    res = am.fit(disp='off')
    forecast = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
    if verbose:
        print res.summary()
    return forecast
    
def garch2(returns, maxlb):
    if type(returns) is pd.core.series.Series:
        returns = returns.values

    nrows = returns.shape[0]
    forecast = np.zeros(nrows)
    print 'Adding GARCH...'
    for i in range(maxlb, nrows):
        #print i,returns[0:i+1].shape
        #range nrows+1 is the last index
        am = arch.arch_model(returns[0:i+1]*100)
        #res = am.fit(iter=10)
        res = am.fit(disp='off')
        forecast[i] = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * \
                    res.resid[i]**2 + res.conditional_volatility[i]**2 * res.params['beta[1]'])
    print res.summary()
    return forecast
    
def autocorrel(close, lb, period=1):
    if type(close) is pd.core.series.Series:
        close = close.values

    ac = np.insert(ta.CORREL(close[0:len(close)-period], close[period:len(close)], timeperiod=lb),0,np.zeros(period))
    return np.nan_to_num(ac)
    
def kaufman_efficiency(close,lb):
    if type(close) is pd.core.series.Series:
        close = close.values
        
    ke = np.nan_to_num(abs(ta.CMO(close, timeperiod=lb)/100))
    return ke

def directionalVolumeSpike(volume, p, lb):
    # current volume / avg volume (* price chg week ago.)
    if type(volume) is pd.core.series.Series:
        close = volume.values
        
    nrows = p.shape[0]
    r = np.zeros(nrows)
    dvs = np.zeros(nrows)
    for i in range(lb, nrows):
        r[i] = (p[i]-p[i-lb])/p[i-lb]
        dvs[i] = volume[i]/np.mean(volume[i-lb:i])*r[i]
    return dvs

def smoothHurst(p,bars):
    if type(p) is pd.core.series.Series:
        p = close.values

    bars=30
    nrows=p.shape[0]
    Dimen=np.zeros(nrows)
    Hurst=np.zeros(nrows)
    SmoothHurst=np.zeros(nrows)

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
    return SmoothHurst
    
def hurst(ts):
    #create range of lag values
    lags = range(2,len(ts))
    #calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    #use a linear fit to estimate the hurt exponent
    poly = polyfit(log(lags), log(tau),1)
    #return the hurst exponent from the polyfit output
    return poly[0]*2.0
 
    
def volumeSpike(volume, lb):
    # current volume / avg volume (* price chg week ago.)
    if type(volume) is pd.core.series.Series:
        close = volume.values
    nrows = volume.shape[0]
    #r = np.zeros(nrows)
    vs = np.zeros(nrows)
    for i in range(lb, nrows):
        #r[i] = (p[i]-p[i-lb])/p[i-lb]
        vs[i] = volume[i]/np.mean(volume[i-lb:i])#*r[i]
    return vs
    
def RSI(p,lb):
    # RSI technical indicator.
    # p, the series having its RSI computed.
    # lb, the lookback period, does not need to be integer.
    #     typical values in the range of 1.5 to 5.0.
    # Return is a numpy array with values in the range 0.0 to 1.0.
    nrows = p.shape[0]
    lam = 2.0 / (lb + 1.0)
    UpMove = np.zeros(nrows)
    DnMove = np.zeros(nrows)
    UpMoveSm = np.zeros(nrows)
    DnMoveSm = np.zeros(nrows)
    Numer = np.zeros(nrows)
    Denom = np.zeros(nrows)
    pChg = np.zeros(nrows)
    RSISeries = np.zeros(nrows)
    # Compute pChg in points using a loop.
    for i in range (1,nrows):
        pChg[i] = p[i] - p[i-1]    
    # Compute pChg as a percentage using a built-in method.
#    pChg = p.pct_change()
    UpMove = np.where(pChg>0,pChg,0)
    DnMove = np.where(pChg<0,-pChg,0)
    
    for i in range(1,nrows):
        UpMoveSm[i] = lam*UpMove[i] + (1.0-lam)*UpMoveSm[i-1]
        DnMoveSm[i] = lam*DnMove[i] + (1.0-lam)*DnMoveSm[i-1]
        Numer[i] = UpMoveSm[i]
        Denom[i] = UpMoveSm[i] + DnMoveSm[i]
        if Denom[i] <= 0:
            RSISeries[i] = 0.5
        else:
            RSISeries[i] =  Numer[i]/Denom[i]
    return(RSISeries)
#  -------------------------------

def ROC(p,lb):
    # Rate of change technical indicator.
    # p, the series having its ROC computed.
    # lb, the lookback period.  Typically 1.
    # Return is a numpy array with values as decimal fractions.
    # A 1% change is 0.01.
    nrows = p.shape[0]
    r = np.zeros(nrows)
    for i in range(lb, nrows):
        r[i] = (p[i]-p[i-lb])/p[i-lb]
    return(r)
#  ----------------------------------

def zScore(p,lb):
    # z score statistic.
    # p, the series having its z-score computed.
    # lb, the lookback period, an integer.
    #     the length used for the average and standard deviation.
    #     typical values 3 to 10.
    # Return is a numpy array with values as z-scores centered on 0.0.
    nrows = p.shape[0]
    st = np.zeros(nrows)
    ma = np.zeros(nrows)
    # use the pandas sliding window functions.
    st = pd.rolling_std(p,lb)
    ma = pd.rolling_mean(p,lb)
    z = np.zeros(nrows)
    for i in range(lb,nrows):
        z[i] = (p[i]-ma[i])/st[i]
    return(z)
#  ----------------------------------

def softmax(p,lb,lam):
    # softmax transformation.
    # p, the series being transformed.
    # lb, the lookback period, an integer.
    #     the length used for the average and standard deviation.
    #     typical values 20 to 252.  Be aware of ramp-up requirement.
    # lam, the length of the linear section.
    #     in standard deviations.
    #     typical value is 6.
    # Return is a numpy array with values in the range 0.0 to 1.0.
    nrows = p.shape[0]
    a = np.zeros(nrows)
    ma = np.zeros(nrows)
    sd = np.zeros(nrows)    
    sm = np.zeros(nrows)
    sq = np.zeros(nrows)
    y = np.zeros(nrows)
    for i in range(lb,nrows):
        sm[i] = sm[i]+p[i]
    ma[i] = sm[i] / lb
    for i in range(lb,nrows):
        sq[i] = (p[i]-ma[i])*(p[i]-ma[i])
    sd[i] = math.sqrt(sq[i]/(nrows-1))
    for i in range(lb,nrows):
        a[i] = (p[i]-ma[i])/((lam*sd[i])/(2.0*math.pi))
        y[i] = 1.0 / (1.0 + math.e**a[i])
    return(y)
    
#  -------------------------------    
    
def softmax_score(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out
    
def DPO(p,lb):
    # Detrended price oscillator. 
    # A high pass filter.
    # p, the series being transformed.
    # lb, the lookback period, a real number.
    # Uses pandas ewma function.
    # Return is a numpy array with values centered on 0.0.
    nrows = p.shape[0]
    ma = pd.ewma(p,span=lb)
    d = np.zeros(nrows)
    for i in range(1,nrows):
        d[i] = (p[i]-ma[i])/ma[i]
    return(d)
#  ----------------------------------
    
def numberZeros(p):
    # Counts the number of zero crossings.
    # p, the series being counted.
    # Return is an integer.  
    # The number of times p crosses up through 0.0. 
    nrows = p.shape[0]
    st = 0
    for i in range(1,nrows):
        if (p[i-1]<0 and p[i]>=0):
            st = st+1
    lt = 0
    for i in range(1,nrows):
        if (p[i-1]>0 and p[i]<=0):
            lt = lt+1
    return(st,lt)
    
def gainAhead(p,lookforward=1):
    if type(p) is pd.core.series.Series:
        p = p.values
    # Computes change in the next 1 bar.
    # p, the base series.
    # Return is a numpy array of changes.
    # A change of 1% is 0.01
    # The final value is unknown.  Its value is 0.0.
    nrows = p.shape[0]
    g = np.zeros(nrows)
    for i in range(0,nrows-lookforward):
        g[i] = (p[i+lookforward]-p[i])/p[i]
    return(g)
    
def ATR(ph,pl,pc,lb):
    # Average True Range technical indicator.
    # ph, pl, pc are the series high, low, and close.
    # lb, the lookback period.  An integer number of bars.
    # True range is computed as a fraction of the closing price.
    # Return is a numpy array of floating point values.
    # Values are non-negative, with a minimum of 0.0.
    # An ATR of 5 points on a issue closing at 50 is
    #    reported as 0.10. 
    nrows = pc.shape[0]
    th = np.zeros(nrows)
    tl = np.zeros(nrows)
    tc = np.zeros(nrows)
    tr = np.zeros(nrows)
    trAvg = np.zeros(nrows)
    
    for i in range(1,nrows):
        if ph[i] > pc[i-1]:
            th[i] = ph[i]
        else:
            th[i] = pc[i-1]
        if pl[i] < pc[i-1]:
            tl[i] = pl[i]
        else:
            tl[i] = pc[i-1]
        tr[i] = th[i] - tl[i]
    for i in range(lb,nrows):
        trAvg[i] = tr[i]            
        for j in range(1,lb-1):
            trAvg[i] = trAvg[i] + tr[i-j]
        trAvg[i] = trAvg[i] / lb
        trAvg[i] = trAvg[i] / pc[i]    
    return(trAvg)
    
def ATR2(ph,pl,pc,lb):
    #not as %of close
    # Average True Range technical indicator.
    # ph, pl, pc are the series high, low, and close.
    # lb, the lookback period.  An integer number of bars.
    # True range is computed as a fraction of the closing price.
    # Return is a numpy array of floating point values.
    # Values are non-negative, with a minimum of 0.0.

    nrows = pc.shape[0]
    th = np.zeros(nrows)
    tl = np.zeros(nrows)
    tc = np.zeros(nrows)
    tr = np.zeros(nrows)
    trAvg = np.zeros(nrows)
    
    for i in range(1,nrows):
        if ph[i] > pc[i-1]:
            th[i] = ph[i]
        else:
            th[i] = pc[i-1]
        if pl[i] < pc[i-1]:
            tl[i] = pl[i]
        else:
            tl[i] = pc[i-1]
        tr[i] = th[i] - tl[i]
    for i in range(lb,nrows):
        trAvg[i] = tr[i]            
        for j in range(1,lb-1):
            trAvg[i] = trAvg[i] + tr[i-j]
        trAvg[i] = trAvg[i] / lb
        trAvg[i] = trAvg[i]
    return(trAvg)
    
#  ----------------------------------

def priceChange(p):
    nrows = p.shape[0]
    pc = np.zeros(nrows)
    for i in range(1,nrows):
        pc[i] = (p[i]-p[i-1])/p[i-1]
    return pc
    
def runsZScore(ga,lb):
    # Computes runs test

    nrows = ga.shape[0]
    #convert gainAhead into binary
    ytrue = [-1 if x<0 else 1 for x in ga]
    rt = np.zeros(nrows)
    for i in range(lb,nrows):
        zs = runstest_1samp(ytrue[i-lb:i])[0]
        #how to find neg inf in series and change to zero
        #dataSet.runsScore10.ix[np.where(np.isinf(dataSet['runsScore10'].values))] =0
        if zs == -np.inf:
            zs = -3
        elif zs == np.inf:
            zs = 3
            
        rt[i] = zs
    return(rt)

def percentUpDays(ga,lb):
    # Computes runs test

    nrows = ga.shape[0]
    #convert gainAhead into binary
    ytrue = [0 if x<0 else 1 for x in ga]
    rt = np.zeros(nrows)
    for i in range(lb,nrows):
        rt[i] = sum(ytrue[i-lb:i])/float(lb)
    return(rt)
    
#  -------------------------------------------------------   
def getToxCutoff(p, datapoints):
    # TOP 25 volatility days
    if datapoints > p.shape[0]:
        #print 'ToxCutoff:',datapoints,'min datapoints >',p.shape[0],'rows. returning',p.shape[0],'rows.'
        return p.shape[0], 0
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    days = 0
    cutoff = 100
    while days < datapoints:
        cutoff = cutoff -5        
        days = int(round(len(X2)*(100-cutoff)/100,0))
        threshold = round(stats.scoreatpercentile(X2,cutoff),6)        
        #print days, cutoff, threshold
    return days, threshold
    
def getToxCutoff2(p, cutoff):
    # TOP 25 volatility days
    #if datapoints > p.shape[0]:
    #    #print 'ToxCutoff:',datapoints,'min datapoints >',p.shape[0],'rows. returning',p.shape[0],'rows.'
    #    return p.shape[0], 0
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    days = int(round(len(X2)*(cutoff/100.0),0))
    threshold = round(stats.scoreatpercentile(X2,100-cutoff),6)        
    #print days, cutoff, threshold        
    return days, threshold
    
def getToxCutoff2(p, cutoff):
    # TOP 25 volatility days
    #if datapoints > p.shape[0]:
    #    #print 'ToxCutoff:',datapoints,'min datapoints >',p.shape[0],'rows. returning',p.shape[0],'rows.'
    #    return p.shape[0], 0
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    days = int(round(len(X2)*(cutoff/100.0),0))
    threshold = round(stats.scoreatpercentile(X2,100-cutoff),6)        
    #print days, cutoff, threshold        
    return days, threshold
    
def create_indicators(qt):
    RSILookback = 1.5
    zScoreLookback = 10
    ATRLookback = 5
    beLongThreshold = 0.0
    DPOLookback = 3    
    #takes dc phase 82 rows to ramp up/prime/spit out the first number.   
    maxlb = 82
    
    #transform functions workonly with numerical indexes starting from zero 
    unfilteredData = pd.concat([qt[' OPEN'],qt[' HIGH'],qt[' LOW'],qt[' CLOSE'],qt[' VOL']], axis=1)
    unfilteredData.columns = ['Open','Low','High','Close','Volume']
    
    Open = unfilteredData.Open.values
    low = unfilteredData.Low.values
    high = unfilteredData.High.values
    close = unfilteredData.Close.values
    volume = unfilteredData.Volume.values
    
    unfilteredData['pctChangeOpen'] = priceChange(Open)
    unfilteredData['pctChangeLow'] = priceChange(low)
    unfilteredData['pctChangeHigh'] = priceChange(high)
    #transform relative to close
    #PRICE LEVEL
    unfilteredData['linreg'] = zScore(ta.LINEARREG(close, timeperiod=7)/close,zScoreLookback)
    unfilteredData['tsf'] = zScore(ta.TSF(close, timeperiod=7)/close,zScoreLookback)
    unfilteredData['support'], unfilteredData['resistance']= ta.MINMAX(close, timeperiod=10)/close
    unfilteredData['support']=zScore(unfilteredData['support'],zScoreLookback)
    unfilteredData['resistance']=zScore(unfilteredData['resistance'],zScoreLookback)
    unfilteredData['bbupper'], unfilteredData['bbmiddle'], unfilteredData['bblower'] = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)/close
    unfilteredData['bbupper']=zScore(unfilteredData['bbupper'],zScoreLookback)
    unfilteredData['bbmiddle']=zScore(unfilteredData['bbmiddle'],zScoreLookback)
    unfilteredData['bblower']=zScore(unfilteredData['bblower'],zScoreLookback)
    
    # transform to zs
    #MOMENTUM
    unfilteredData['roc'] = zScore(ta.ROC(close, timeperiod=2),20)
    unfilteredData['rocp'] = zScore(ta.ROCP(close, timeperiod=3),20)
    unfilteredData['rocr'] = zScore(ta.ROCR(close, timeperiod=4),20)
    unfilteredData['rocr100'] = zScore(ta.ROCR100(close, timeperiod=5),20)
    unfilteredData['trix'] = zScore(ta.TRIX(close, timeperiod=2),20)
    unfilteredData['ultosc'] = zScore(ta.ULTOSC(high, low, close, timeperiod1=3, timeperiod2=5, timeperiod3=10),20)
    unfilteredData['willr'] = zScore(ta.WILLR(high, low, close, timeperiod=7),20)
    unfilteredData['linreg_slope'] = zScore(ta.LINEARREG_SLOPE(close, timeperiod=7),20)
    unfilteredData['adx'] = zScore(ta.ADX(high, low, close, timeperiod=5),20)
    unfilteredData['adxr'] = zScore(ta.ADXR(high, low, close, timeperiod=3),20)
    unfilteredData['macd'], unfilteredData['macdsignal'], unfilteredData['macdhist'] = ta.MACD(close, fastperiod=3, slowperiod=7, signalperiod=3)
    unfilteredData['macd'] = zScore(unfilteredData['macd'],20)
    unfilteredData['macdsignal'] = zScore(unfilteredData['macdsignal'],20)
    unfilteredData['macdhist'] = zScore(unfilteredData['macdhist'],20)
    unfilteredData['minus_di'] = zScore(ta.MINUS_DI(high, low, close, timeperiod=7),20)
    unfilteredData['minus_dm'] = zScore(ta.MINUS_DM(high, low, timeperiod=7),20)
    unfilteredData['cmo'] = zScore(ta.CMO(close, timeperiod=3),20)
    unfilteredData['mom'] = zScore(ta.MOM(close, timeperiod=3),20)
    unfilteredData['plus_di'] = zScore(ta.PLUS_DI(high, low, close, timeperiod=5),20)
    unfilteredData['plus_dm'] = zScore(ta.PLUS_DM(high, low, timeperiod=5),20)
    unfilteredData['ppo'] = zScore(ta.PPO(close, fastperiod=5, slowperiod=10, matype=0),20)
    #Volume
    #unfilteredData['ad'] = zScore(ta.AD(high, low, close, volume),20)
    #unfilteredData['adosc'] = zScore(ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10),20)
    unfilteredData['obv'] = zScore(ta.OBV(close, volume),20)
    #VOLATILITY
    unfilteredData['atr'] = zScore(ta.ATR(high, low, close, timeperiod=7),20)
    unfilteredData['natr'] = zScore(ta.NATR(high, low, close, timeperiod=10),20)
    unfilteredData['trange'] = zScore(ta.TRANGE(high, low, close),20)
    #CYCLE
    unfilteredData['ht_dcperiod'] = zScore(ta.HT_DCPERIOD(close),20)
    unfilteredData['ht_dcphase'] = zScore(ta.HT_DCPHASE(close),20)
    unfilteredData['inphase'], unfilteredData['quadrature'] = ta.HT_PHASOR(close)
    unfilteredData['inphase']=zScore(unfilteredData['inphase'],20)
    unfilteredData['quadrature']=zScore(unfilteredData['quadrature'],20)
    
    # div 100
    #MOMENTUM
    unfilteredData['apo'] = ta.APO(close, fastperiod=3, slowperiod=10, matype=0)/100.0
    unfilteredData['aroondown'], unfilteredData['aroonup'] = ta.AROON(high, low, timeperiod=7)
    unfilteredData['aroondown']=unfilteredData['aroondown']/100.0
    unfilteredData['aroonup']=unfilteredData['aroonup']/100.0
    unfilteredData['aroonosc'] = ta.AROONOSC(high, low, timeperiod=5)/100.0
    unfilteredData['cci'] = ta.CCI(high, low, close, timeperiod=5)/100.0
    unfilteredData['dx'] = ta.DX(high, low, close, timeperiod=5)/100.0
    unfilteredData['mfi'] = ta.MFI(high, low, close, volume, timeperiod=7)/100.0
    unfilteredData['rsi'] = ta.RSI(close, timeperiod=14)/100.0
    unfilteredData['slowk'], unfilteredData['slowd'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    unfilteredData['slowk']=unfilteredData['slowk']/100.0
    unfilteredData['slowd']=unfilteredData['slowd']/100.0
    unfilteredData['fastk'], unfilteredData['fastd'] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    unfilteredData['fastk']=unfilteredData['fastk']/100.0
    unfilteredData['fastd']=unfilteredData['fastd']/100.0
    unfilteredData['rsifastk'], unfilteredData['rsifastd'] = ta.STOCHRSI(close, timeperiod=10, fastk_period=5, fastd_period=3, fastd_matype=0)
    unfilteredData['rsifastk']=unfilteredData['rsifastk']/100.0
    unfilteredData['rsifastd']=unfilteredData['rsifastd']/100.0
    
    #no transform
    unfilteredData['bop'] = ta.BOP(Open, high, low, close)
    #CYCLE
    unfilteredData['sine'], unfilteredData['leadsine'] = ta.HT_SINE(close)
    unfilteredData['ht_trendmode'] = ta.HT_TRENDMODE(close)
             
    #unfilteredData.to_csv('/media/sf_Python/indicators.csv')
    # sanity check the file
    #pd.concat([filteredData,unfilteredData], axis=1, join='outer').to_csv('/media/sf_Python/indexcheck_outer_join.csv')
    #pd.concat([filteredData,unfilteredData], axis=1, join='inner').to_csv('/media/sf_Python/indexcheck_inner_join.csv')
    
    #checks if there are nan/inf values beyond the first maxlb rows
    for col in unfilteredData:
        if sum(np.isnan(unfilteredData[col][maxlb:].values))>0:
            print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
            raise ValueError, 'nan in %s' % col
        elif sum(np.isinf(unfilteredData[col][maxlb:].values))>0:
            print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
            raise ValueError, 'inf in %s' % col
        elif sum(np.isneginf(unfilteredData[col][maxlb:].values))>0:
            print unfilteredData[col][maxlb:][np.isnan(unfilteredData[col][maxlb:].values)]
            raise ValueError, '-inf in %s' % col
            
    return unfilteredData
  
#  -------------------------------------------------------    
