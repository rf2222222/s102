# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tick
import seaborn as sns
import datetime
from os import listdir
from os.path import isfile, join
from pytz import timezone
from datetime import datetime as dt
from scipy import stats
from scipy.stats import kurtosis, skew
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import statsmodels.tsa.stattools as ts
from sklearn.preprocessing import scale, robust_scale, minmax_scale
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,\
                            log_loss,precision_score,recall_score, roc_auc_score,\
                            confusion_matrix, hamming_loss, jaccard_similarity_score,\
                            zero_one_loss
from transform import softmax_score, numberZeros, ratio, hurst


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.color_palette("Set1", n_colors=10, desat=.5)

def offlineMode(ticker, errorText, signalPath, ver1, ver2):
        files = [ f for f in listdir(signalPath) if isfile(join(signalPath,f)) ]
        if ver1+'_'+ ticker + '.csv' in files:
            signalFile=pd.read_csv(signalPath+ ver1+'_'+ ticker + '.csv', parse_dates=['dates'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.signals = 0
            offline.gainAhead =0
            offline.prior_index=0
            offline.safef=0
            offline.CAR25=0
            offline.dd95 = 0
            offline.ddTol=0
            offline.system = errorText
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + ver1+'_'+ ticker + '.csv', index=False)
            
        if ver2+'_'+ ticker + '.csv' in files:
            signalFile=pd.read_csv(signalPath+ ver2+'_'+ ticker + '.csv', parse_dates=['dates'])
            offline = signalFile.iloc[-1].copy(deep=True)
            offline.dates = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.signals = 0
            offline.gainAhead =0
            offline.prior_index=0
            offline.safef=0
            offline.CAR25=0
            offline.dd95 = 0
            offline.ddTol=0
            offline.system = errorText
            offline.timestamp = str(pd.to_datetime(dt.now(timezone('US/Eastern')).replace(second=0, microsecond=0)))[:-6]
            offline.cycleTime = 0
            signalFile=signalFile.append(offline)
            signalFile.to_csv(signalPath + ver2+'_'+ ticker + '.csv', index=False)
        print errorText    
        #sys.exit(errorText)
        
def describeDistribution2(qtC,ticker):
    hold_days = 1
    #qtC = dataSet.reset_index()['Close']
    #qtP = dataSet.reset_index()['priceChange']
    nrows = qtC.shape[0]
    qtP=qtC.pct_change().fillna(0)
    # ------------------------
    #   Set up gainer and loser lists
    gainer = np.zeros(nrows, dtype=int)
    loser = np.zeros(nrows, dtype=int)
    i_gainer = 0
    i_loser = 0

    for i in range(0,nrows-hold_days):
        if (qtC[i+hold_days]>qtC[i]):
            gainer[i_gainer] = i
            i_gainer = i_gainer + 1
        else:
            loser[i_loser] = i
            i_loser = i_loser + 1
    number_gainers = i_gainer
    number_losers = i_loser
    
    hist, bins = np.histogram(qtP.values, bins=100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='r')
    plt.title('____Price Distribution____')
    plt.show()
    kurt = kurtosis(qtP.values)
    sk = skew(qtP.values)
    print 'Issue:', ticker,
    #print 'Dates:             ' + str(qtC.index[0])
    #print '  to:              ' + str(qtC.index[-1])
    print 'Rows:', str(qtC.shape[0]),
    #print 'Cols:              ' + str(qtC.shape[1])
    print 'Number Gainers: ', number_gainers,
    print 'Number Losers: ', number_losers
    print 'Mean: ', qtP.values.mean(), ' StDev: ',qtP.values.std()
    print ' Kurtosis: ', kurt, ' Skew: ', sk


    #X2 = np.sort(qtP)
    #F2 = np.array(range(qtP.shape[0]), dtype=float)/qtP.shape[0]
    #plt.title( '____Cumulative Distribution____')
    #plt.plot(F2,X2, color='r')
    #plt.show()

    #tox  = getToxCDF(abs(qtP), display=True)

    #entropy
    ent = 0
    hist = hist[np.nonzero(hist)].astype(float)
    for i in hist/sum(hist):
        ent -= i * math.log(i, len(hist))
        #print i,ent
    print 'Entropy: ', ent

    
def describeDistribution(qtC,qtP,ticker):
    hold_days = 1
    #qtC = dataSet.reset_index()['Close']
    #qtP = dataSet.reset_index()['priceChange']
    nrows = qtC.shape[0]

    # ------------------------
    #   Set up gainer and loser lists
    gainer = np.zeros(nrows, dtype=int)
    loser = np.zeros(nrows, dtype=int)
    i_gainer = 0
    i_loser = 0

    for i in range(0,nrows-hold_days):
        if (qtC[i+hold_days]>qtC[i]):
            gainer[i_gainer] = i
            i_gainer = i_gainer + 1
        else:
            loser[i_loser] = i
            i_loser = i_loser + 1
    number_gainers = i_gainer
    number_losers = i_loser
    
    hist, bins = np.histogram(qtP.values, bins=100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='r')
    plt.title('____Price Distribution____')
    plt.show()
    
    print 'Issue:', ticker,
    #print 'Dates:             ' + str(qtC.index[0])
    #print '  to:              ' + str(qtC.index[-1])
    print 'Rows:', str(qtC.shape[0]),
    #print 'Cols:              ' + str(qtC.shape[1])
    print 'Number Gainers: ', number_gainers,
    print 'Number Losers: ', number_losers
    
    kurt = kurtosis(qtP.values)
    sk = skew(qtP.values)
    print 'Mean: ', qtP.values.mean(), ' StDev: ',qtP.values.std()
    print ' Kurtosis: ', kurt, ' Skew: ', sk


    X2 = np.sort(qtP)
    F2 = np.array(range(qtP.shape[0]), dtype=float)/qtP.shape[0]
    plt.title( '____Cumulative Distribution____')
    plt.plot(F2,X2, color='r')
    plt.show()

    tox  = getToxCDF(abs(qtP), display=True)

    #entropy
    ent = 0
    hist = hist[np.nonzero(hist)].astype(float)
    for i in hist/sum(hist):
        ent -= i * math.log(i, len(hist))
        #print i,ent
    print '\nEntropy:              ', ent

    #ADF, Hurst
    adf_test(qtC)
    
def adf_test(series):
    adf = ts.adfuller(series,1)
    #create a GBM, MR, Trending series
    gbm = np.log(np.cumsum(np.random.randn(100000))+1000)
    mr = np.log(np.random.randn(100000)+1000)
    tr = np.log(np.cumsum(np.random.randn(100000)+1)+1000)
    
    print ""
    print "ADF test for mean reversion"
    #print "Datapoints", adf[3]
    #print "p-value", adf[1]
    print "Test-Stat", adf[0]
    for key in adf[4]:
     print "Critical Values:",key, adf[4][key],
     if adf[0] < adf[4][key]:
         print 'PASS'
     else:
         print 'FAIL'
    print ""
    print "Hurst Exponent Test"
    print "Hurst(Random Walk): %s" % hurst(gbm)
    print "Hurst(MR): %s" % hurst(mr)
    print "Hurst(Trend): %s" % hurst(tr)
    print "Hurst(series): %s" % hurst(series)

    
def showDistribution():
    step=250
    for i in range(0,len(qt),step):
        if i+step>len(qt)-1:
            end = len(qt)-1
        else:
            end = i+step
        print '\n\n', qt.index[i], ' to ', qt.index[end]
        hist, bins = np.histogram(qt.gainAhead.values[i:end], bins=100)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width, color='r')
        plt.show()
        kurt = kurtosis(qt.gainAhead.values[i:end])
        sk = skew(qt.gainAhead.values[i:end])
        runs = runstest_1samp(qt.gainAhead.values[i:end])[0]
        print 'Mean: ', qt.gainAhead.values[i:end].mean(), 'StDev: ' ,qt.gainAhead.values[i:end].std()
        print 'Kurtosis: ', kurt, ' Skew: ', sk, 'Runs: ', runs
        
def displayRankedCharts(numCharts,benchmarks,benchStatsByYear,equityCurves,equityStats,equityCurvesStatsByYear,\
                                            **kwargs):
    vsDPS=kwargs.get('vsDPS',False)
    dpsRank=kwargs.get('dpsRank',None)
    dpsChartRank=kwargs.get('dpsChartRank',0)
    yscale=kwargs.get('yscale','log')
    v3tag=kwargs.get('v3tag',None)
    savePath=kwargs.get('savePath',None)
    showPlot=kwargs.get('showPlot',True)
    verbose=kwargs.get('verbose',True)
    
    topSystem = equityStats.sort_values(['scoremm'], ascending=False).system.iloc[0]
    leftoverIndex = equityStats.shape[0]%numCharts
    eIndex = range(equityStats.shape[0]-numCharts,-numCharts,-numCharts)
    dpsChartRank = dpsChartRank + 1
    tt_index =[]

    # show charts numCharts at a time..
    for i in eIndex:
        #last wf index adjust the test index, else step
        if leftoverIndex > 0 and i == eIndex[-1]:
            show_index = range(0,leftoverIndex)        
            tt_index.insert(0,show_index)
            #print i, show_index
        else:
            show_index = range(i,numCharts+i)
            tt_index.insert(0,show_index)
            #print i, show_index
            
    for tti in tt_index:
        #if top system (rank 0) used as benchmark, exclude rank 0
        #if 0 in tti and topSystem in benchmarks:
        #    tti.pop(0)
            
        topSystems = equityStats.sort_values(['scoremm'], ascending=False).system.iloc[tti]
        chartRank = [x+1 for x in tti]
        #init plots
        fig = plt.figure(1)#1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        fig2 = plt.figure(2)#, figsize=(6, 6))
        ax2 = fig2.add_subplot(111)
            
        #plot top systems
        for i,sst in enumerate(topSystems): 
            nrows = equityCurves[sst].shape[0]
            #  Plot the equitycurve and drawdown
            if not equityCurves[sst].index.to_datetime()[0].time() and not equityCurves[sst].index.to_datetime()[-1].time():
                barSize = '1 day'
                
                def format_date(x, pos=None):
                    thisind = np.clip(int(x + 0.5), 0, nrows - 1)
                    return equityCurves[sst].index[thisind].strftime("%Y-%m-%d")
            
                #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
                #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
                
            else:
                barSize = '1 min'
                
                def format_date(x, pos=None):
                    thisind = np.clip(int(x + 0.5), 0, nrows - 1)
                    return equityCurves[sst].index[thisind].strftime("%Y-%m-%d %H:%M")
                    
                #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
                #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
                
            minorLocator = MultipleLocator(nrows)
            ax.xaxis.set_minor_locator(minorLocator)
            ax2.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
            ax2.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
            xticks = ax.xaxis.get_minor_ticks()
            xticks[1].label1.set_visible(False)
            xticks = ax2.xaxis.get_minor_ticks()
            xticks[1].label1.set_visible(False)
            
            ind_ec = np.arange(nrows)
            plt.figure(1)
            if vsDPS:
                ax.plot(ind_ec, equityCurves[sst].equity, label=str(dpsChartRank)+'_'+sst)
            else:
                ax.plot(ind_ec, equityCurves[sst].equity, label=str(chartRank[i])+'_'+sst)
            #plt.subplot(2,1,2)
            plt.figure(2)
            if vsDPS:
                ax2.plot(ind_ec, -equityCurves[sst].drawdown, label=str(dpsChartRank)+'_'+sst)
            else:
                ax2.plot(ind_ec, -equityCurves[sst].drawdown, label=str(chartRank[i])+'_'+sst)

        #plot benchmarks
        for sf1 in benchmarks: 
            #plt.subplot(2,1,1)
            if 'sellHold' in sf1:
                color='lightpink'
            elif 'buyHold' in sf1:
                color='c'
            else:
                color='lightgreen'

            plt.figure(1)
            ax.set_xlim(0, benchmarks[sf1].shape[0])
            ind_bm = np.arange(benchmarks[sf1].shape[0])
            ax.plot(ind_bm, benchmarks[sf1].equity,'--', label=sf1,color=color)
            if vsDPS:
                plt.title('Rank '+str(dpsChartRank)+' SAFEF1 vs DPS')
            else:
                plt.title(v3tag+' Top '+str(chartRank).strip('[]')+' Systems')
            plt.ylabel('Equity')
            y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.grid('on')
            ax.set_yscale(yscale)
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.2),prop={'size':10})
            #lgd=ax.legend(loc="upper left",prop={'size':10})
            fig.autofmt_xdate()
            #fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
            #plt.subplot(2,1,2)
            plt.figure(2)
            ax2.set_xlim(0, benchmarks[sf1].shape[0])
            ax2.plot(ind_bm, -benchmarks[sf1].drawdown,'--', label=sf1,color=color)
            ax2.yaxis.set_major_formatter(y_formatter)
            plt.ylabel('Drawdown')
            ax2.grid('on')
            handles, labels = ax2.get_legend_handles_labels()
            #lgd2=plt.legend(loc="upper left",prop={'size':10})
            lgd2 = ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.2),prop={'size':10})
            #fig2.savefig('samplefigure', bbox_extra_artists=(lgd2,), bbox_inches='tight')
            fig2.autofmt_xdate()
        
        if showPlot:
            plt.show()
        plt.close(fig2)
        
        if savePath != None:
            plt.figure(1)
            #shortTrades, longTrades = numberZeros(equityCurves[topSystem].signals.values)
            signals = equityCurves[topSystem].signals * equityCurves[topSystem].safef.astype(int)
            allTrades = sum((signals).round().diff().fillna(0).values !=0)
            hoursTraded = (equityCurves[topSystem].index[-1]-equityCurves[topSystem].index[0]).total_seconds()/60.0/60.0
            avgTrades = float(allTrades)/hoursTraded
            
            #text= '\nValidation Period from', topSystem.index[0],'to',topSystem.index[-1]
            text=  '\n%0.f bar counts: ' % signals.shape[0]
            #if 1 in signals.value_counts():
            text+= '%i beLong,  ' % signals[signals>0].shape[0]
            #if -1 in SST.signals.value_counts():
            text+= '%i beShort,  ' % signals[signals<0].shape[0]
            #if 0 in SST.signals.value_counts():
            text+= '%i beFlat  ' % signals[signals==0].shape[0]
            text='\nAverage trades per hour: %0.2f' % (avgTrades)
            #text+='\nTWR for Buy & Hold is %0.3f, %i Bars' % (equityBuyHold[nrows-1], nrows)
            #text+='\nTWR for Sell & Hold is %0.3f, %i Bars' % (equitySellHold[nrows-1], nrows)
            #text+='\nTWR for %i beLong trades is %0.3f' % (longTrades, equityBeLongSignals[nrows-1])
            #text+='\nTWR for %i beShort trades is %0.3f' % (shortTrades,equityBeShortSignals[nrows-1])
            text+='\nTWR for %i trades with DPS is %0.3f' % (allTrades,equityCurves[topSystem].equity[-1]) 
            if 'avgSafef' in equityStats:
                    text+='\nAvg. safef: %.3f' % equityStats.loc[equityStats.system == topSystem].avgSafef
            plt.figtext(0.05,-0.2,text, fontsize=15)
            print 'Saving '+savePath+v3tag+'.png'
            fig.savefig(savePath+v3tag+'.png', bbox_inches='tight')
                
        plt.close(fig)
        
                
        if verbose:
            for sf1 in benchmarks:    
                print '\nbenchmark: ', sf1
                #if sf1 == 'buyHoldSafef1':
                #    print 'Signal: None'
                #else:
                #    print 'Signal: ', sf1
                startDate = benchmarks[sf1].index.to_datetime()[0]
                endDate = benchmarks[sf1].index.to_datetime()[-1]
                yearsInValidation = (endDate-startDate).total_seconds()/3600.0/365.0
                print 'Validation Length (years): %.2f' % yearsInValidation
                print 'In the market (Bars): %i' % benchmarks[sf1].numBars.iloc[-1],
                shortTrades, longTrades = numberZeros(benchmarks[sf1].signals*benchmarks[sf1].safef)
                allTrades = sum((benchmarks[sf1].signals * benchmarks[sf1].safef).round().diff().fillna(0).values !=0)
                hoursTraded = (endDate-startDate).total_seconds()/60.0/60.0
                TPY = allTrades/hoursTraded
                print ' AvgTrades/Hr: %.2f' % TPY,
                CAR =100*(((benchmarks[sf1].equity.iloc[-1]/benchmarks[sf1].equity.iloc[0])**(1.0/yearsInValidation))-1.0)
                print '\nCAR: %.3f' % CAR,
                MAXDD = max(benchmarks[sf1].maxDD)*-100.0
                print 'maxDD: %.1f%%' % MAXDD,
                print 'Sortino: %.3f ' %  ratio(benchmarks[sf1].equity).sortino(),
                print 'Sharpe: %.3f ' % ratio(benchmarks[sf1].equity).sharpe()
                marRatio = CAR/-MAXDD
                slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(benchmarks[sf1].equity.values)),benchmarks[sf1].equity.values)
                k_ratio = (slope/std_err) * math.sqrt(252.0)/len(benchmarks[sf1].equity.values)
                print 'K-Ratio: %.3f ' %  k_ratio,
                print 'MAR: %.3f ' % marRatio
                print benchStatsByYear[sf1]
            
            #maxCAR = -np.inf
            for i,sst in enumerate(topSystems): 
                #dayAhead = (endDate+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                #signalAhead = equityCurves[sst].signals[-1]
                if 'CAR25' not in equityCurves[sst].columns:
                    safefAhead = equityStats[equityStats.system==sst].safef.iloc[-1]
                    CAR25ahead = equityStats[equityStats.system==sst].CAR25.iloc[-1]
                    #dd100 as an approx for dd95 because forecast period is only 2 years for df_CAR25 calc
                    dd95Ahead = equityStats[equityStats.system==sst].DD100.iloc[-1]        
                else:
                    safefAhead = equityCurves[sst].safef.iloc[-1]
                    CAR25ahead = equityCurves[sst].CAR25.iloc[-1]
                    dd95Ahead = equityCurves[sst].dd95.iloc[-1]
                
                print '\n',str(chartRank[i]),sst
                print 'In the Market (Bars): %i' % equityCurves[sst].numBars.iloc[-1],
                shortTrades, longTrades = numberZeros(equityCurves[sst].signals*equityCurves[sst].safef)
                allTrades = sum((equityCurves[sst].signals * equityCurves[sst].safef).round().diff().fillna(0).values !=0)
                hoursTraded = (equityCurves[sst].index[-1]-equityCurves[sst].index[0]).total_seconds()/60.0/60.0
                TPY = allTrades/hoursTraded
                print ' AvgTrades/Hr: %.2f' % TPY
                if 'avgSafef' in equityStats:
                    print 'Avg. safef: %.3f' % equityStats.loc[equityStats.system == sst].avgSafef,
                print 'CAR: %.3f ' % equityStats.loc[equityStats.system == sst].cumCAR,
                if 'CAR25' in equityStats:
                    print 'CAR25: %.3f ' % equityStats.loc[equityStats.system == sst].CAR25, 
                if 'CAR50' in equityStats:
                    print 'CAR50: %.3f ' % equityStats.loc[equityStats.system == sst].CAR50, 
                if 'CAR75' in equityStats:
                    print 'CAR75: %.3f ' % equityStats.loc[equityStats.system == sst].CAR75
                print 'maxDD: %.1f%% ' % equityStats.loc[equityStats.system == sst].MAXDD
                print 'Sortino: %.3f ' %  equityStats.loc[equityStats.system == sst].sortinoRatio,           
                print 'Sharpe: %.3f ' % equityStats.loc[equityStats.system == sst].sharpeRatio,
                print 'K-Ratio: %.3f ' %  equityStats.loc[equityStats.system == sst].k_ratio,
                print 'MAR: %.3f ' % equityStats.loc[equityStats.system == sst].marRatio
                print equityCurvesStatsByYear[sst]
                #print dayAhead, ' Signal:', equityCurves[sst].signals[-1],
                #print 'safef: %.3f ' % safefAhead, 'CAR25: %.3f ' % CAR25ahead,
                #print 'dd95:%.3f ' % dd95Ahead
                
            #save sst with max car    
            #if CAR > maxCAR:
            #    maxCAR = CAR
            #    maxCARsst = sst
            
           
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    if scoring is not None:
        plt.ylabel(scoring + " Score")
    else:
        plt.ylabel("Auto-Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def compareEquity_vf(sst, title):
    nrows = sst.gainAhead.shape[0]
    #signalCounts = sst.signals.shape[0]
    
    print '\nValidation Period from', sst.index[0],'to',sst.index[-1]
    print '\nThere are %0.f signal counts' % nrows
    if 1 in sst.signals.value_counts():
        print sst.signals.value_counts()[1], 'High Volatility Signals',
    if -1 in sst.signals.value_counts():
        print sst.signals.value_counts()[-1], 'Low Volatility Signals'
    #if 0 in sst.signals.value_counts():
    #    print sst.signals.value_counts()[0], 'beFlat Signals',
    #  Compute cumulative equity for all days
    equityAllSignals = np.zeros(nrows)
    equityAllSignals[0] = 1
    for i in range(1,nrows):
        equityAllSignals[i] = (1+sst.gainAhead.iloc[i-1])*equityAllSignals[i-1]
        
    #  Compute cumulative equity for days with HV signals    
    equityHVLong = np.zeros(nrows)
    equityHVLong[0] = 1
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] > 0):
            equityHVLong[i] = (1+sst.gainAhead.iloc[i-1])*equityHVLong[i-1]
        else:
            equityHVLong[i] = equityHVLong[i-1]
            
    #  Compute cumulative equity for days with LV signals    
    equityLVLong = np.zeros(nrows)
    equityLVLong[0] = 1
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] < 0):
            equityLVLong[i] = (1+sst.gainAhead.iloc[i-1])*equityLVLong[i-1]
        else:
            equityLVLong[i] = equityLVLong[i-1]

    #  Compute cumulative equity for days with beShort signals    
    equityHVShort = np.zeros(nrows)
    equityHVShort[0] = 1
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] > 0):
            equityHVShort[i] = (1+-sst.gainAhead.iloc[i-1])*equityHVShort[i-1]
        else:
            equityHVShort[i] = equityHVShort[i-1] 

    equityLVShort = np.zeros(nrows)
    equityLVShort[0] = 1
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] < 0):
            equityLVShort[i] = (1+-sst.gainAhead.iloc[i-1])*equityLVShort[i-1]
        else:
            equityLVShort[i] = equityLVShort[i-1]
            
    LVtrades, HVtrades = numberZeros(sst.signals.values)
    allTrades = LVtrades+ HVtrades        
    #plt.close('all')
    fig = plt.figure(1)#1, figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(sst.index.to_datetime(), equityHVLong,label="Long HV Signals",color='b')
    ax1.plot(sst.index.to_datetime(), equityHVShort,label="Short HV Signals",color='r')
    ax1.plot(sst.index.to_datetime(), equityAllSignals,label="BuyHold",ls='--',color='c')
    fig.autofmt_xdate()
    fig.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.title('High Volatility '+title)
    plt.ylabel("TWR")
    plt.legend(loc="best")
    plt.show()
    print 'TWR for Buy & Hold is %0.3f, %i days' % (equityAllSignals[nrows-1], nrows)
    print 'TWR for %i HV Long trades is %0.3f' % (HVtrades, equityHVLong[nrows-1])
    print 'TWR for %i HV Short trades is %0.3f' % (HVtrades,equityHVShort[nrows-1])
    
    fig2 = plt.figure(2)#, figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(sst.index.to_datetime(), equityLVLong,label="Long LV Signals",color='b')   
    ax2.plot(sst.index.to_datetime(), equityLVShort,label="Short LV Signals",color='r')
    ax2.plot(sst.index.to_datetime(), equityAllSignals,label="BuyHold",ls='--',color='c')
    #ax.plot(sst.index.to_datetime(), equityBeLongAndShortSignals,label="Long & Short",color='m')  
    # rotate and align the tick labels so they look better
    fig2.autofmt_xdate()
    fig2.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.title('Low Volatility '+title)
    plt.ylabel("TWR")
    plt.legend(loc="best")
    plt.show()
    print 'TWR for Buy & Hold is %0.3f, %i days' % (equityAllSignals[nrows-1], nrows)
    print 'TWR for %i LV Long trades is %0.3f' % (LVtrades, equityLVLong[nrows-1])
    print 'TWR for %i LV Short trades is %0.3f' % (LVtrades,equityLVShort[nrows-1])
    #print 'TWR for %i beLong and beShort trades is %0.3f' % (allTrades,equityBeLongAndShortSignals[nrows-1])
    
    #check
    #pd.concat([sst,pd.Series(data=equityAllSignals,name='equityAllSignals',index=sst.index),\
    #        pd.Series(data=equityHVLong,name='equityHVLong',index=sst.index),
    #        pd.Series(data=equityBeShortSignals,name='equityBeShortSignals',index=sst.index),
    #        pd.Series(data=equityBeLongAndShortSignals,name='equityBeLongAndShortSignals',index=sst.index),
    #        ],axis=1)
    
def compareEquity(sst, title, **kwargs):
    savePath = kwargs.get('savePath',None)
    version= kwargs.get('version',None)
    showChart = kwargs.get('showChart',True)
    ticker =  kwargs.get('ticker',None)
    filename =  kwargs.get('filename',None)
    initialEquity =  kwargs.get('initialEquity',1)
    #check if there's time in the index
    if not sst.index.to_datetime()[0].time() and not sst.index.to_datetime()[1].time():
        barSize = '1 day'
    else:
        barSize = '1 min'
        
    nrows = sst.gainAhead.shape[0]
    #signalCounts = sst.signals.shape[0]
    print '\nThere are %0.f signal counts' % nrows
    if 1 in sst.signals.value_counts():
        print sst.signals.value_counts()[1], 'beLong Signals',
    if -1 in sst.signals.value_counts():
        print sst.signals.value_counts()[-1], 'beShort Signals',
    if 0 in sst.signals.value_counts():
        print sst.signals.value_counts()[0], 'beFlat Signals',
    #  Compute cumulative equity for all days
    equityBuyHold = np.zeros(nrows)
    equityBuyHold[0] = initialEquity
    for i in range(1,nrows):
        equityBuyHold[i] = (1+sst.gainAhead.iloc[i-1])*equityBuyHold[i-1]
        
    equitySellHold = np.zeros(nrows)
    equitySellHold[0] = initialEquity
    for i in range(1,nrows):
        equitySellHold[i] = (1-sst.gainAhead.iloc[i-1])*equitySellHold[i-1]
        
    #  Compute cumulative equity for days with beLong signals    
    equityBeLongSignals = np.zeros(nrows)
    equityBeLongSignals[0] = initialEquity
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] > 0):
            equityBeLongSignals[i] = (1+sst.gainAhead.iloc[i-1])*equityBeLongSignals[i-1]
        else:
            equityBeLongSignals[i] = equityBeLongSignals[i-1]
            
    #  Compute cumulative equity for days with beShort signals    
    equityBeLongAndShortSignals = np.zeros(nrows)
    equityBeLongAndShortSignals[0] = initialEquity
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] < 0):
            equityBeLongAndShortSignals[i] = (1+-sst.gainAhead.iloc[i-1])*equityBeLongAndShortSignals[i-1]
        elif (sst.signals.iloc[i-1] > 0):
            equityBeLongAndShortSignals[i] = (1+sst.gainAhead.iloc[i-1])*equityBeLongAndShortSignals[i-1]
        else:
            equityBeLongAndShortSignals[i] = equityBeLongAndShortSignals[i-1]

    #  Compute cumulative equity for days with beShort signals    
    equityBeShortSignals = np.zeros(nrows)
    equityBeShortSignals[0] = initialEquity
    for i in range(1,nrows):
        if (sst.signals.iloc[i-1] < 0):
            equityBeShortSignals[i] = (1+-sst.gainAhead.iloc[i-1])*equityBeShortSignals[i-1]
        else:
            equityBeShortSignals[i] = equityBeShortSignals[i-1] 
    
    #plt.close('all')
    fig, ax = plt.subplots(1, figsize=(8,7))
    ind = np.arange(nrows)
    ax.plot(ind, equityBeLongSignals,label="Long 1 Signals",color='b')
    ax.plot(ind, equityBeShortSignals,label="Short -1 Signals",color='r')
    ax.plot(ind, equityBeLongAndShortSignals,label="Long & Short",color='g')
    ax.plot(ind, equityBuyHold,label="BuyHold",ls='--',color='c')
    ax.plot(ind, equitySellHold,label="SellHold",ls='--',color='lightpink')
    # rotate and align the tick labels so they look better
    #years = mdates.YearLocator()   # every year
    #months = mdates.MonthLocator()  # every month
    #days = mdates.DayLocator()
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    if barSize != '1 day' :
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, nrows - 1)
            return sst.index[thisind].strftime("%Y-%m-%d %H:%M")
            
        #hours = mdates.HourLocator() 
        #minutes = mdates.MinuteLocator()
        # format the ticks
        #ax.xaxis.set_minor_locator(minutes)
        #ax.xaxis.set_major_locator(hours)
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
    else:
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, nrows - 1)
            return sst.index[thisind].strftime("%Y-%m-%d")
            
        # format the ticks
        #ax.xaxis.set_major_locator(years)
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #ax.xaxis.set_minor_locator(months)        
        # format the ticks
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        #ax.xaxis.set_minor_locator(years)
        
    ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    minorLocator = MultipleLocator(nrows)
    ax.xaxis.set_minor_formatter(tick.FuncFormatter(format_date))
    ax.xaxis.set_minor_locator(minorLocator)
    xticks = ax.xaxis.get_minor_ticks()
    xticks[1].label1.set_visible(False)
    
    ax.set_xlim(0, nrows)
    plt.title(title)
    plt.ylabel("TWR")
    plt.legend(loc="best")
    fig.autofmt_xdate()
    
    shortTrades, longTrades = numberZeros(sst.signals.values)
    allTrades = shortTrades+ longTrades
    hoursTraded = (sst.index[-1]-sst.index[0]).total_seconds()/60.0/60.0
    avgTrades = float(allTrades)/hoursTraded
    
    if savePath != None:
        if filename==None:
            plt.savefig(savePath+title+'.png', bbox_inches='tight')
        else:
            text=  '\n%0.f bar counts: ' % nrows
            if 1 in sst.signals.value_counts():
                text+= '%i beLong,  ' % sst.signals.value_counts()[1]
            if -1 in sst.signals.value_counts():
                text+= '%i beShort,  ' % sst.signals.value_counts()[-1]
            if 0 in sst.signals.value_counts():
                text+= '%i beFlat  ' % sst.signals.value_counts()[0]
            #text= '\nValidation Period from', sst.index[0],'to',sst.index[-1]
            text='\nAverage trades per hour: %0.2f' % (avgTrades)
            #text+='\nTWR for Buy & Hold is %0.3f, %i Bars' % (equityBuyHold[nrows-1], nrows)
            #text+='\nTWR for Sell & Hold is %0.3f, %i Bars' % (equitySellHold[nrows-1], nrows)
            text+='\nTWR for %i beLong trades is %0.3f' % (longTrades, equityBeLongSignals[nrows-1])
            text+='\nTWR for %i beShort trades is %0.3f' % (shortTrades,equityBeShortSignals[nrows-1])
            text+='\nTWR for %i trades is %0.3f' % (allTrades,equityBeLongAndShortSignals[nrows-1])   
            plt.figtext(0.05,-0.0,text, fontsize=15)
            print 'Saving '+savePath+filename+'.png'
            plt.savefig(savePath+filename+'.png', bbox_inches='tight')
            
    if showChart:        
        plt.show()
    plt.close(fig)
    
        
    #shortTrades, longTrades = numberZeros(sst.signals.values)
    #allTrades = shortTrades+ longTrades
    #hoursTraded = (sst.index[-1]-sst.index[0]).total_seconds()/60.0/60.0
    #avgTrades = float(allTrades)/hoursTraded
    print '\nValidation Period from', sst.index[0],'to',sst.index[-1]
    print 'Average trades per hour: %0.2f' % (avgTrades)
    print 'TWR for Buy & Hold is %0.3f, %i Bars' % (equityBuyHold[nrows-1], nrows)
    print 'TWR for Sell & Hold is %0.3f, %i Bars' % (equitySellHold[nrows-1], nrows)
    print 'TWR for %i beLong trades is %0.3f' % (longTrades, equityBeLongSignals[nrows-1])
    print 'TWR for %i beShort trades is %0.3f' % (shortTrades,equityBeShortSignals[nrows-1])
    print 'TWR for %i beLong and beShort trades is %0.3f' % (allTrades,equityBeLongAndShortSignals[nrows-1])
    
    
    #check
    #pd.concat([sst,pd.Series(data=equityBuyHold,name='equityBuyHold',index=sst.index),\
    #        pd.Series(data=equityBeLongSignals,name='equityBeLongSignals',index=sst.index),
    #        pd.Series(data=equityBeShortSignals,name='equityBeShortSignals',index=sst.index),
    #        pd.Series(data=equityBeLongAndShortSignals,name='equityBeLongAndShortSignals',index=sst.index),
    #        ],axis=1)
    
def getToxCDF(p, display=True):
    # TOP 25 volatility days
    X2 = np.sort(p)
    F2 = np.array(range(p.shape[0]), dtype=float)/p.shape[0]
    TOX05  =round(stats.scoreatpercentile(X2,95),6)
    TOX10 = round(stats.scoreatpercentile(X2,90),6)
    TOX15 = round(stats.scoreatpercentile(X2,85),6)
    TOX20 = round(stats.scoreatpercentile(X2,80),6)
    TOX25 = round(stats.scoreatpercentile(X2,75),6)
    if display:
        plt.title( '____Volatility Distribution____')
        t25 = np.empty_like(X2)
        t25.fill(TOX25)
        plt.plot(F2,X2,'b-',F2, t25,'r--')
        plt.show()
        
        print "TOX05: ", TOX05, 'Days:', round(len(X2)*0.05,1)
        print "TOX10: ", TOX10, 'Days:', round(len(X2)*0.10,1)
        print "TOX15: ", TOX15, 'Days:', round(len(X2)*0.15,1)
        print "TOX20: ", TOX20, 'Days:', round(len(X2)*0.20,1)
        print "TOX25: ", TOX25, 'Days:', round(len(X2)*0.25,1)
    return {'TOX05':TOX05,'TOX10':TOX10,'TOX15':TOX15,'TOX20':TOX20,'TOX25':TOX25}

def showPDF(ytrue, ypred, gainAhead, index, **kwargs):
    savePath = kwargs.get('savePath',None)
    filename = kwargs.get('filename',None)
    showPDFCDF = kwargs.get('showPDFCDF',True)
    
    n_bins=50
    tp_index = index[np.intersect1d(np.where(ypred == 1), np.where(ytrue == 1))]
    fn_index = index[np.intersect1d(np.where(ypred == -1), np.where(ytrue == 1))]
    fp_index = index[np.intersect1d(np.where(ypred == 1), np.where(ytrue == -1))]
    tn_index = index[np.intersect1d(np.where(ypred == -1), np.where(ytrue == -1))]
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flat

    if tp_index.shape[0] > 1:
        ax0.hist(np.array(gainAhead.iloc[tp_index]*100), n_bins, histtype='bar', color="g")
    ax0.set_title('TP %i F1 %.2f '% (tp_index.shape[0], f1_score(ytrue, ypred)))
    if fn_index.shape[0] > 1:
        ax1.hist(np.array(gainAhead.iloc[fn_index]*100), n_bins, histtype='bar', color="r")
    ax1.set_title('FN %i Recall %.2f '% (fn_index.shape[0], recall_score(ytrue, ypred)))
    if fp_index.shape[0] > 1:
        ax2.hist(np.array(gainAhead.iloc[fp_index]*100), n_bins, histtype='bar', color="r")
    ax2.set_title('FP %i Precision %.2f '% (fp_index.shape[0], precision_score(ytrue, ypred)))
    if tn_index.shape[0] > 1:
        ax3.hist(np.array(gainAhead.iloc[tn_index]*100), n_bins, histtype='bar', color="g")
    ax3.set_title('TN %i Accuracy %.2f '% (tn_index.shape[0], accuracy_score(ytrue, ypred)))
    #hist, bins = np.histogram(gainAhead[tp_index], bins=100)
    #width = 0.7 * (bins[1] - bins[0])
    #center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)
    plt.tight_layout()
    if savePath != None and filename != None:
        plt.savefig(savePath+filename+'_PDF.png', bbox_inches='tight')
        
    if showPDFCDF:
        print '                          ____Price Distribution____'
        plt.show()
    plt.close()

def showCDF(ytrue, ypred, gainAhead, index, **kwargs):
    savePath = kwargs.get('savePath',None)
    filename = kwargs.get('filename',None)
    showPDFCDF = kwargs.get('showPDFCDF',True)
    
    tp_index = index[np.intersect1d(np.where(ypred == 1), np.where(ytrue == 1))]
    fn_index = index[np.intersect1d(np.where(ypred == -1), np.where(ytrue == 1))]
    fp_index = index[np.intersect1d(np.where(ypred == 1), np.where(ytrue == -1))]
    tn_index = index[np.intersect1d(np.where(ypred == -1), np.where(ytrue == -1))]
    tpfp_index = np.concatenate((tp_index,fp_index))
    fntn_index = np.concatenate((fn_index,tn_index))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    #ax0, ax1, ax2, ax3 = axes.flat
    ax0, ax1 = axes.flat
    
    F = np.array(range(gainAhead.iloc[tpfp_index].shape[0]), dtype=float)/gainAhead.iloc[tpfp_index].shape[0]
    ax0.plot(F,pd.Series.sort_values(gainAhead.iloc[tpfp_index]*100), color="b")
    split = float(tpfp_index.shape[0])/float((tpfp_index.shape[0]+fntn_index.shape[0]))
    if float(tpfp_index.shape[0]) != 0:
        acc = float(tp_index.shape[0])/float(tpfp_index.shape[0])
    else:
        acc = 0.0
    ax0.set_title('1: TP+FP %i, Split: %.2f, Acc:  %.2f'% (tpfp_index.shape[0],split,acc))
    F = np.array(range(gainAhead.iloc[fntn_index].shape[0]), dtype=float)/gainAhead.iloc[fntn_index].shape[0]
    ax1.plot(F,pd.Series.sort_values(gainAhead.iloc[fntn_index]*100), color="b")
    split = float(fntn_index.shape[0])/float((tpfp_index.shape[0]+fntn_index.shape[0]))
    if float(fntn_index.shape[0]) != 0:
        acc = float(tn_index.shape[0])/float(fntn_index.shape[0])
    else:
        acc = 0.0
    ax1.set_title('-1: FN+TN %i, Split %.2f, Acc:  %.2f'% (fntn_index.shape[0],split,acc))
    #F = np.array(range(gainAhead.iloc[fp_index].shape[0]), dtype=float)/gainAhead.iloc[fp_index].shape[0]
    #ax2.plot(F,pd.Series.sort_values(gainAhead.iloc[fp_index]*100))
    #ax2.set_title('FP %i Precision %.2f '% (fp_index.shape[0], precision_score(ytrue, ypred)))
    #F = np.array(range(gainAhead.iloc[tn_index].shape[0]), dtype=float)/gainAhead.iloc[tn_index].shape[0]
    #ax3.plot(F,pd.Series.sort_values(gainAhead.iloc[tn_index]*100))
    #ax3.set_title('TN %i Accuracy %.2f '% (tn_index.shape[0], accuracy_score(ytrue, ypred)))

    
    #plt.plot(F2,X2)
    plt.tight_layout()
    if savePath != None and filename != None:
        plt.savefig(savePath+filename+'_CDF.png', bbox_inches='tight')
        
    if showPDFCDF:
        print '                       ____Cumulative Distribution____'
        plt.show()
    plt.close()
        
def display_CAR25(CAR25):
    print 'Signal: ', CAR25['C25sig'], ' ', CAR25['Type']
    print 'Fcst Horizon (years): %.1f ' % CAR25['YIF'], 
    print ' AvgTrades/Yr: %.2f' % CAR25['TPY']
    
    print 'DD95:  %.3f ' % CAR25['DD95'],
    print 'DD100: %.3f ' %  CAR25['DD100'],
    print 'SORTINO25: %.3f ' %  CAR25['SOR25'],
    print 'SHARPE25: %.3f ' % CAR25['SHA25']
    
    print 'SAFEf: %.3f ' % CAR25['safef'],
    print 'CAR25: %.2f ' % CAR25['CAR25'],
    print 'CAR50: %.2f ' % CAR25['CAR50'],
    print 'CAR75: %.2f ' % CAR25['CAR75']
    
#def init_report_prospector():
#    col = ['ticker','sample','signal','start','end','rows','accuracy',\
#            'C25sig', 'safef', 'CAR25', 'CAR50', 'CAR75', 'DD95', 'DD100', 'SOR25', 'SHA25', 'YIF', 'TPY',\
#            'test_split','tox_adj']
#    return pd.DataFrame(columns=col)
def validation_scoring(model_metrics, sample1):
    #create scores, find and save best model
    #removed f1, rec, rows, to focus more on profitability (magnitude)
    #removed sortino - equity curve scoring is not factored in here
    #Only Minmax score of the separated samples 1 and 2 are used to rank between apples and apples.
    #removed accuracy and drawdown to focus more on monte carlo distribution, added TPY to minimize trades
    #sample1
    model_score_s1 = model_metrics.loc[model_metrics['sample'] == sample1].reset_index().copy(deep=True)
    #model_score_s1['f1mm'] =minmax_scale(robust_scale(model_score_s1.f1.reshape(-1, 1)))
    #model_score_s1['accmm'] =minmax_scale(robust_scale(model_score_s1.acc.reshape(-1, 1)))
    #model_score_s1['precmm'] =minmax_scale(robust_scale(model_score_s1.prec.reshape(-1, 1)))
    #model_score_s1['recmm'] = minmax_scale(robust_scale(model_score_s1.rec.reshape(-1, 1)))
    #model_score_s1['fn_magmm'] =-minmax_scale(robust_scale(model_score_s1.fn_mag.reshape(-1, 1)))
    #model_score_s1['fp_magmm'] =-minmax_scale(robust_scale(model_score_s1.fp_mag.reshape(-1, 1)))
    model_score_s1['CAR25mm'] =minmax_scale(robust_scale(model_score_s1.CAR25.reshape(-1, 1)))
    model_score_s1['CAR50mm'] =minmax_scale(robust_scale(model_score_s1.CAR50.reshape(-1, 1)))
    model_score_s1['CAR75mm'] =minmax_scale(robust_scale(model_score_s1.CAR75.reshape(-1, 1)))
    #model_score_s1['DD100mm'] =-minmax_scale(robust_scale(model_score_s1.DD100.reshape(-1, 1)))
    #model_score_s1['SOR25mm'] =minmax_scale(robust_scale(model_score_s1.SOR25.reshape(-1, 1)))
    #model_score_s1['TPYmm'] =-minmax_scale(robust_scale(model_score_s1.TPY.reshape(-1, 1)))
    #model_score_s1['rowsmm'] = minmax_scale(robust_scale(model_score_s1.rows.reshape(-1, 1)))   
    model_score_s1['scoremm'] =    model_score_s1.CAR25mm+\
                                    model_score_s1.CAR50mm+model_score_s1.CAR75mm
                                    #model_score_s1.fn_magmm+model_score_s1.fp_magmm+\
                                    #model_score_s1.TPYmm
                                    #model_score_s1.DD100mm
                                    #model_score_s1.accmm+
                                    #+model_score_s1.SOR25mm+

    scored_models = model_score_s1.sort_values(['scoremm'], ascending=False).drop(['index'], axis=1)
    bestModel = scored_models.iloc[0]

    return scored_models, bestModel
    
def directional_scoring(model_metrics, sample1, sample2=None):
    #create scores, find and save best model
    #removed f1, rec, rows, to focus more on profitability (magnitude)
    #removed sortino - equity curve scoring is not factored in here
    #Only Minmax score of the separated samples 1 and 2 are used to rank between apples and apples.
    #removed accuracy and drawdown to focus more on monte carlo distribution, added TPY to minimize trades
    #sample1
    model_score_s1 = model_metrics.loc[model_metrics['sample'] == sample1].reset_index().copy(deep=True)
    model_score_s1['f1mm'] =minmax_scale(robust_scale(model_score_s1.f1.reshape(-1, 1)))
    model_score_s1['accmm'] =minmax_scale(robust_scale(model_score_s1.acc.reshape(-1, 1)))
    model_score_s1['precmm'] =minmax_scale(robust_scale(model_score_s1.prec.reshape(-1, 1)))
    model_score_s1['recmm'] = minmax_scale(robust_scale(model_score_s1.rec.reshape(-1, 1)))
    model_score_s1['fn_magmm'] =-minmax_scale(robust_scale(model_score_s1.fn_mag.reshape(-1, 1)))
    model_score_s1['fp_magmm'] =-minmax_scale(robust_scale(model_score_s1.fp_mag.reshape(-1, 1)))
    model_score_s1['CAR25mm'] =minmax_scale(robust_scale(model_score_s1.CAR25.reshape(-1, 1)))
    model_score_s1['CAR50mm'] =minmax_scale(robust_scale(model_score_s1.CAR50.reshape(-1, 1)))
    model_score_s1['CAR75mm'] =minmax_scale(robust_scale(model_score_s1.CAR75.reshape(-1, 1)))
    #model_score_s1['DD100mm'] =-minmax_scale(robust_scale(model_score_s1.DD100.reshape(-1, 1)))
    model_score_s1['SOR25mm'] =minmax_scale(robust_scale(model_score_s1.SOR25.reshape(-1, 1)))
    model_score_s1['TPYmm'] =-minmax_scale(robust_scale(model_score_s1.TPY.reshape(-1, 1)))
    #model_score_s1['rowsmm'] = minmax_scale(robust_scale(model_score_s1.rows.reshape(-1, 1)))   
    model_score_s1['scoremm'] =    model_score_s1.CAR25mm+\
                                    model_score_s1.CAR50mm+model_score_s1.CAR75mm+\
                                    model_score_s1.fn_magmm+model_score_s1.fp_magmm+\
                                    model_score_s1.TPYmm+model_score_s1.SOR25mm
                                    #model_score_s1.DD100mm
                                    #model_score_s1.accmm+
                                    
    if sample2 is not None:
        #sample2                        
        model_score_s2 = model_metrics.loc[model_metrics['sample'] == sample2].reset_index()
        model_score_s2['f1mm'] =minmax_scale(robust_scale(model_score_s2.f1.reshape(-1, 1)))
        model_score_s2['accmm'] =minmax_scale(robust_scale(model_score_s2.acc.reshape(-1, 1)))
        model_score_s2['precmm'] =minmax_scale(robust_scale(model_score_s2.prec.reshape(-1, 1)))
        model_score_s2['recmm'] = minmax_scale(robust_scale(model_score_s2.rec.reshape(-1, 1)))
        model_score_s2['fn_magmm'] =-minmax_scale(robust_scale(model_score_s2.fn_mag.reshape(-1, 1)))
        model_score_s2['fp_magmm'] =-minmax_scale(robust_scale(model_score_s2.fp_mag.reshape(-1, 1)))
        model_score_s2['CAR25mm'] =minmax_scale(robust_scale(model_score_s2.CAR25.reshape(-1, 1)))
        model_score_s2['CAR50mm'] =minmax_scale(robust_scale(model_score_s2.CAR50.reshape(-1, 1)))
        model_score_s2['CAR75mm'] =minmax_scale(robust_scale(model_score_s2.CAR75.reshape(-1, 1)))
        model_score_s2['DD100mm'] =-minmax_scale(robust_scale(model_score_s2.DD100.reshape(-1, 1)))
        #model_score_s2['SOR25mm'] =minmax_scale(robust_scale(model_score_s2.SOR25.reshape(-1, 1)))
        #model_score_s2['TPYmm'] =minmax_scale(robust_scale(model_score_s2.TPY.reshape(-1, 1)))
        #model_score_s2['rowsmm'] = minmax_scale(robust_scale(model_score_s2.rows.reshape(-1, 1)))
        model_score_s2['scoremm'] =  model_score_s2.accmm+model_score_s2.CAR25mm+\
                                    model_score_s2.CAR50mm+model_score_s2.CAR75mm+\
                                        model_score_s2.fn_magmm+model_score_s2.fp_magmm+\
                                        model_score_s2.DD100mm#+model_score_s2.SOR25mm+model_score_s2.TPYmm
        
        #stack the sample1 scores and sample 2 scores. directional scoring is straightforward so softmax score is not used
        s2_scoremm = pd.concat([model_score_s2['scoremm'],model_score_s2['scoremm']], axis=0)
        s2_scoremm.name = sample2+'scoremm'
        s1_scoremm = pd.concat([model_score_s1['scoremm'],model_score_s1['scoremm']], axis=0)
        s1_scoremm.name = sample1+'scoremm'
        #minimum of both scores to find best model
        combined_final =pd.concat([s2_scoremm,s1_scoremm], axis=1).min(axis=1)
        combined_final.name = 'final_score'
        c_score_df = pd.concat([combined_final, s2_scoremm, s1_scoremm], axis=1)

        #save metrics
        scored_models = pd.concat([c_score_df, pd.concat([model_score_s2,model_score_s1], axis=0)],axis=1)\
                    .sort_values(['final_score'], ascending=False).drop(['index'], axis=1)
        bestModel = model_score_s2.iloc[combined_final.idxmax()]
    else:
        scored_models = model_score_s1.sort_values(['scoremm'], ascending=False).drop(['index'], axis=1)
        bestModel = scored_models.iloc[0]
    
    return scored_models, bestModel
                
def update_report_prospector(original_report, sample, CAR25, metaData):
    report =  init_report()
    report['sample'] = pd.Series(sample)
    
    for item in metaData:
        report[item] = pd.Series(metaData[item])
    for item in CAR25:
        report[item] = pd.Series(CAR25[item])
    #print report
    #print pd.concat([original_report,report])
    return pd.concat([original_report,report])
    
def init_report():
    #col = ['ticker','model','params','sample','signal','date__start','date_end','f1','acc','prec','rec','fn_mag', 'fp_mag',\
    #        'C25sig', 'safef', 'CAR25', 'CAR50', 'CAR75', 'DD95', 'DD100', 'SOR25', 'SHA25', 'YIF', 'TPY',\
    #        'avg_prec','logloss','ham','tp','fn','fp','tn','rows','cols','test_split','iters','tox_adj']
    return pd.DataFrame()#columns=col)
    
def update_report(original_report, sample, ypred, ytrue, gainAhead, index, m, metaData, CAR25=[]):
    report = init_report()
    report['model'] = pd.Series(str(m[0]))
    report['params'] = pd.Series(str(m[1]))
    report['sample'] = pd.Series(sample)    
    for item in metaData:
        report[item] = pd.Series(metaData[item])
    for item in CAR25:
        report[item] = pd.Series(CAR25[item])
    cm = confusion_matrix(ytrue, ypred)
    if cm.shape != (1,1):
        cm_sum_oos = confusion_matrix(ytrue,ypred).astype(float)
        tp = cm[1,1]
        fn = cm[1,0]
        fp = cm[0,1]
        tn = cm[0,0]
    else:
        if ytrue[0]>0:
            tp = float(len(ytrue))
            tn = 0.0
        else:
            tp = 0.0
            tn = float(len(ytrue))
        fn = 0.0
        fp = 0.0
    fn_index = index[np.intersect1d(np.where(ypred == -1), np.where(ytrue == 1))]
    fp_index = index[np.intersect1d(np.where(ypred == 1), np.where(ytrue == -1))]
    #report['ticker'] = pd.Series(ticker)
    #report['signal'] = pd.Series(signal)
    #report['start'] = pd.Series(FirstYear)
    #report['end'] = pd.Series(FinalYear)
    report['f1'] = pd.Series(f1_score(ytrue, ypred))
    report['acc'] = pd.Series(accuracy_score(ytrue, ypred))
    report['prec'] = pd.Series(precision_score(ytrue, ypred))
    report['rec'] = pd.Series(recall_score(ytrue, ypred))
    if gainAhead.iloc[fn_index].empty:
        report['fn_mag'] = 0
    else:
        report['fn_mag'] = pd.Series(np.mean(np.sqrt(np.array((gainAhead.iloc[fn_index])**2))))
    if gainAhead.iloc[fp_index].empty:
        report['fp_mag'] = 0
    else:        
        report['fp_mag'] = pd.Series(np.mean(np.sqrt(np.array((gainAhead.iloc[fp_index])**2))))
    report['avg_prec'] = pd.Series(average_precision_score(ytrue, ypred))
    report['logloss'] = pd.Series(log_loss(ytrue, ypred))
    #report['roc_auc'] = pd.Series(roc_auc_score(ytrue, ypred))
    report['ham'] = pd.Series(hamming_loss(ytrue, ypred))
    #report['jacc'] = pd.Series(jaccard_similarity_score(ytrue, ypred))
    #report['01loss'] = pd.Series(zero_one_loss(ytrue, ypred))
    report['tp'] = pd.Series(tp)
    report['fn'] = pd.Series(fn)
    report['fp'] = pd.Series(fp)
    report['tn'] = pd.Series(tn)
    

    #print report
    #print pd.concat([original_report,report])
    return pd.concat([original_report,report])
    
def is_display_cmatrix2(ytrue, ypred, gainAhead, index, m, ticker, testFirstYear, testFinalYear, iterations, signal, show=0):
    print "\nSymbol is ", ticker
    print "Learning algorithm is ", m
    print "Signal is ", signal
    #print "Confusion matrix for %i randomized tests for %i rows" % (iterations, ytrue.shape[0])
    print "for years ", testFirstYear, " through ", testFinalYear 
    
    print "\nIn sample"
    if show==1:
        showPDF(ytrue, ypred, gainAhead, index)
        showCDF(ytrue, ypred, gainAhead, index)
    else:
        cm_sum_is = confusion_matrix(ytrue,ypred).astype(float)
        tpIS = cm_sum_is[1,1]
        fnIS = cm_sum_is[1,0]
        fpIS = cm_sum_is[0,1]
        tnIS = cm_sum_is[0,0]
        precisionIS = tpIS/(tpIS+fpIS)
        recallIS = tpIS/(tpIS+fnIS)
        accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
        f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 
        print "      pos neg"
        print "pos:  %i  %i  %.2f" % (tpIS, fnIS, recallIS)
        print "neg:  %i  %i" % (fpIS, tnIS)
        print "      %.2f          %.2f " % (precisionIS, accuracyIS)
        print "f1:   %.2f" % f1IS


def oos_display_cmatrix2(ytrue, ypred, gainAhead, index, m, ticker,testFirstYear,\
                                                testFinalYear, iterations, signal, **kwargs):
    showPDFCDF=kwargs.get('showPDFCDF',True)
    savePath=kwargs.get('savePath',None)
    filename=kwargs.get('filename',None)
    verbose=kwargs.get('verbose',None)
    if verbose:
        print "\nSymbol is ", ticker
        print "Learning algorithm is ", m
        print "Signal is ", signal
        #print "Confusion matrix for %i randomized tests for %i rows" % (iterations, ytrue.shape[0])
        print "for years ", testFirstYear, " through ", testFinalYear 
        
        print "\nOut of sample"
    if showPDFCDF == True or (savePath != None and filename != None):
        showPDF(ytrue, ypred, gainAhead, index,showPDFCDF=showPDFCDF,\
                        savePath=savePath,filename=filename)
        showCDF(ytrue, ypred, gainAhead, index,showPDFCDF=showPDFCDF,\
                        savePath=savePath,filename=filename)
    else:
        if confusion_matrix(ytrue,ypred).shape != (1,1):
            cm_sum_oos = confusion_matrix(ytrue,ypred).astype(float)
            tpOOS = cm_sum_oos[1,1]
            fnOOS = cm_sum_oos[1,0]
            fpOOS = cm_sum_oos[0,1]
            tnOOS = cm_sum_oos[0,0]
        else:
            if ytrue[0]>0:
                tpOOS = float(len(ytrue))
                tnOOS = 0.0
            else:
                tpOOS = 0.0
                tnOOS = float(len(ytrue))
            fnOOS = 0.0
            fpOOS = 0.0
            
        if (tpOOS+fpOOS) == 0:
            precisionOOS =  np.nan
        else:
            precisionOOS = tpOOS/(tpOOS+fpOOS)
        if (tpOOS+fnOOS) == 0:
            recallOOS =  np.nan
        else:
            recallOOS = tpOOS/(tpOOS+fnOOS)
        if (tpOOS+fnOOS+fpOOS+tnOOS) ==0:
            accuracyOOS =  np.nan
        else:
            accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
        if precisionOOS ==  np.nan or recallOOS ==  np.nan:
            f1OOS =  np.nan
        else:
            f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 

        print "      pos neg"
        print "pos:  %i  %i  %.2f" % (tpOOS, fnOOS, recallOOS)
        print "neg:  %i  %i" % (fpOOS, tnOOS)
        print "      %.2f          %.2f " % (precisionOOS, accuracyOOS)
        print "f1:   %.2f" % f1OOS
    
def is_display_cmatrix(cm_sum_is, m, ticker, testFirstYear, testFinalYear, iterations, signal):
    cm_sum_is = cm_sum_is.astype(float)
    tpIS = cm_sum_is[1,1]
    fnIS = cm_sum_is[1,0]
    fpIS = cm_sum_is[0,1]
    tnIS = cm_sum_is[0,0]
    precisionIS = tpIS/(tpIS+fpIS)
    recallIS = tpIS/(tpIS+fnIS)
    accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
    f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 
    
    print "\nSymbol is ", ticker
    print "Learning algorithm is ", m
    print "Signal is ", signal
    #print "Confusion matrix for %i randomized tests" % iterations
    print "for years ", testFirstYear, " through ", testFinalYear 
    
    print "\nIn sample"
    #print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpIS, fnIS, recallIS)
    print "neg:  %i  %i" % (fpIS, tnIS)
    print "      %.2f          %.2f " % (precisionIS, accuracyIS)
    print "f1:   %.2f" % f1IS

def oos_display_cmatrix(ytrue, ypred, m, ticker,testFirstYear, testFinalYear, iterations, signal):
    #print confusion_matrix(ytrue,ypred).astype(float).shape
    #print confusion_matrix(ytrue,ypred)
    #print ytrue
    #print ypred
    if confusion_matrix(ytrue,ypred).astype(float).shape != (1,1):
        cm_sum_oos = confusion_matrix(ytrue,ypred).astype(float)
        tpOOS = cm_sum_oos[1,1]
        fnOOS = cm_sum_oos[1,0]
        fpOOS = cm_sum_oos[0,1]
        tnOOS = cm_sum_oos[0,0]
    else:
        if ytrue[0]>0:
            tpOOS = float(len(ytrue))
            tnOOS = 0.0
        else:
            tpOOS = 0.0
            tnOOS = float(len(ytrue))
        fnOOS = 0.0
        fpOOS = 0.0
        
    if (tpOOS+fpOOS) == 0:
        precisionOOS =  np.nan
    else:
        precisionOOS = tpOOS/(tpOOS+fpOOS)
    if (tpOOS+fnOOS) == 0:
        recallOOS =  np.nan
    else:
        recallOOS = tpOOS/(tpOOS+fnOOS)
    if (tpOOS+fnOOS+fpOOS+tnOOS) ==0:
        accuracyOOS =  np.nan
    else:
        accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
    if precisionOOS ==  np.nan or recallOOS ==  np.nan:
        f1OOS =  np.nan
    else:
        f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 
        
    print "\nSymbol is ", ticker
    print "Learning algorithm is ", m
    print "Signal is ", signal
    #print "Confusion matrix for %i randomized tests" % iterations
    print "for years ", testFirstYear, " through ", testFinalYear 
    
    print "\nOut of sample"
    #print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpOOS, fnOOS, recallOOS)
    print "neg:  %i  %i" % (fpOOS, tnOOS)
    print "      %.2f          %.2f " % (precisionOOS, accuracyOOS)
    print "f1:   %.2f" % f1OOS
        
        
def sss_display_cmatrix(cm_sum_is, cm_sum_oos, m, ticker,testFirstYear, testFinalYear, iterations, signal):      
    cm_sum_is = cm_sum_is.astype(float)
    cm_sum_oos = cm_sum_oos.astype(float)
    tpIS = cm_sum_is[1,1]
    fnIS = cm_sum_is[1,0]
    fpIS = cm_sum_is[0,1]
    tnIS = cm_sum_is[0,0]
    precisionIS = tpIS/(tpIS+fpIS)
    recallIS = tpIS/(tpIS+fnIS)
    accuracyIS = (tpIS+tnIS)/(tpIS+fnIS+fpIS+tnIS)
    f1IS = (2.0 * precisionIS * recallIS) / (precisionIS+recallIS) 
    
    tpOOS = cm_sum_oos[1,1]
    fnOOS = cm_sum_oos[1,0]
    fpOOS = cm_sum_oos[0,1]
    tnOOS = cm_sum_oos[0,0]
    precisionOOS = tpOOS/(tpOOS+fpOOS)
    recallOOS = tpOOS/(tpOOS+fnOOS)
    accuracyOOS = (tpOOS+tnOOS)/(tpOOS+fnOOS+fpOOS+tnOOS)
    f1OOS = (2.0 * precisionOOS * recallOOS) / (precisionOOS+recallOOS) 
    
    print "\nSymbol is ", ticker
    print "Learning algorithm is ", m
    print "Signal is ", signal
    print "Confusion matrix for %i randomized tests" % iterations
    print "for years ", testFirstYear, " through ", testFinalYear 
    
    print "\nIn sample"
    #print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpIS, fnIS, recallIS)
    print "neg:  %i  %i" % (fpIS, tnIS)
    print "      %.2f          %.2f " % (precisionIS, accuracyIS)
    print "f1:   %.2f" % f1IS
    
    print "\nOut of sample"
    #print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpOOS, fnOOS, recallOOS)
    print "neg:  %i  %i" % (fpOOS, tnOOS)
    print "      %.2f          %.2f " % (precisionOOS, accuracyOOS)
    print "f1:   %.2f" % f1OOS
        
