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



def acPeriodogram(data,bars=48):
    p=data.Close
    nrows=p.shape[0]
    index = p.index
    col_names = ['KeyRev']+['AC_lag'+str(i) for i in range(1,bars)]
    X_train= pd.DataFrame(data=np.zeros([nrows,bars]), index=p.index,\
                            columns=col_names)
    #if lb <3:
    #    print 'lookback < 3. adjusting lookback minimum to 3'
    #    lb =3    
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

    #autoCorr
    for i in range(bars+3,nrows):
        AvgLength=3
        Corr = np.zeros(bars)
        Corr2 = np.zeros([bars,2])
        CosinePart = np.zeros(bars)
        SinePart = np.zeros(bars)
        SqSum = np.zeros(bars)
        R = np.zeros([bars,2])
        Pwr = np.zeros(bars)

        #pearson correlation for each value of lag
        '''
        for Lag in range(0,bars):
            print 'Lag', Lag
            M = AvgLength
            if AvgLength ==0: 
                M = Lag
            Sx=0
            Sy=0
            Sxx=0
            Syy=0
            Sxy=0 
            for count in range(0,M-1):
                print count, Lag
                X = filt[count]
                Y= filt[Lag+count]
                Sx = Sx+Y
                Sxx = Sxx+X*X
                Sxy = Sxy +X*Y
                SYY = Syy + Y*Y
            if (M*Sxx-Sx*Sx)*(M*Syy-Sy*Sy)>0:
                Corr[Lag] = (M*Sxy-Sx*Sy)/math.sqrt((M*Sxx-Sx*Sx)*(M*Syy-Sy*Sy))
        '''
        #need minimum rows of Lag+3
        for Lag in range(0,bars):
            if Lag>0:
                Corr2[Lag,1]=Corr2[Lag-1,0]
            Corr[Lag] = pd.Series(filt[:i]).autocorr(lag=Lag)
            Corr2[Lag,0]=.5*(Corr[Lag]+1)
        '''
        SumDeltas=0
        for Lag in range(3,bars):
            if (Corr2[Lag,0]>.5 and Corr2[Lag,1]<.5) or (Corr2[Lag,0]<.5 and Corr2[Lag,1]>.5):
                #print SumDeltas
                SumDeltas=SumDeltas+1
        #print i, SumDeltas
        
        #Reversal=0
        #if SumDeltas>bars/2:
        #    Reversal=1
        Corr[0]=SumDeltas/float(bars)
        '''
        #print data.index[i]
        if data.iloc[i].Open>data.iloc[i-1].Close and data.iloc[i].Close<data.iloc[i-1].Low:
            #print data.iloc[i].Open,'>' ,data.iloc[i-1].Close, 'and', data.iloc[i].Close,'<',data.iloc[i-1].Low
            Corr[0]=-1
        elif data.iloc[i].Open<data.iloc[i-1].Close and data.iloc[i].Close>data.iloc[i-1].High:
            Corr[0]=1
        else:
            Corr[0]=0
        
        rad370=math.radians(370)
        for Period in range(10,bars):
            CosinePart[Period]=0 
            SinePart[Period]=0
            for N in range(3,bars):
                CosinePart[Period]=CosinePart[Period]+Corr[N]*math.cos(rad370*N/Period)
                SinePart[Period]=SinePart[Period]+Corr[N]*math.sin(rad370*N/Period)
            SqSum[Period]=CosinePart[Period]*CosinePart[Period]+SinePart[Period]*SinePart[Period]

        for Period in range(10,bars):
            R[Period,1]=R[Period-1,0]
            R[Period,0]=.2*SqSum[Period]*SqSum[Period]+.8*R[Period,1]
            
        MaxPwr=0
        MaxPwr = .995*MaxPwr
        for Period in range(10,bars):
            if R[Period,0]>MaxPwr:
                MaxPwr=R[Period,0]

        for Period in range(3,bars):
            Pwr[Period]=R[Period,0]/MaxPwr

        Spx=0
        Sp=0
        for Period in range(10,bars):
            if Pwr[Period]>=.5:
                Spx=Spx+Period*Pwr[Period]
                Sp=Sp+Pwr[Period]

        if Sp != 0:
            DominantCycle=int(Spx/Sp)
        else:
            DominantCycle = None
        #print i, 'dCycle', DominantCycle
        
        #X_train= X_train.append(pd.Series(Corr, name=index[i]))
        
        X_train.set_value(index[i], col_names, Corr)
    return DominantCycle, X_train
    
if __name__ == "__main__":
    p =data.Close
    index = data.Close.index
    gainAhead=data.Close.pct_change().shift(-1).fillna(0)
    models = [#("GA_Reg", SymbolicRegressor(population_size=5000, generations=20,
              #                             tournament_size=20, stopping_criteria=0.0, 
              #                             const_range=(-1.0, 1.0), init_depth=(2, 6), 
              #                             init_method='half and half', transformer=True, 
              #                             comparison=True, trigonometric=True, 
              #                             metric='mean absolute error', parsimony_coefficient=0.001, 
              #                             p_crossover=0.9, p_subtree_mutation=0.01, 
              #                             p_hoist_mutation=0.01, p_point_mutation=0.01, 
              #                             p_point_replace=0.05, max_samples=1.0, 
              #                             n_jobs=1, verbose=0, random_state=None)),
             #("GA_Reg2", SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01, comparison=True, transformer=False, p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1, verbose=0, parsimony_coefficient=0.01, random_state=0)),
             #("LR", LogisticRegression()), \
             #("PRCEPT", Perceptron(class_weight={1:1})), \
             #("PAC", PassiveAggressiveClassifier(class_weight={1:500})), \
             #("LSVC", LinearSVC()), \
             #("GNBayes",GaussianNB()),\
             #("LDA", LinearDiscriminantAnalysis()), \
             #("QDA", QuadraticDiscriminantAnalysis()), \
             #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
             #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
             #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
             #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
             #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
             #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #("RF", RandomForestClassifier(class_weight={1:500}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
             #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
             #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
             #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
             #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
             ("VotingHard", VotingClassifier(estimators=[\
             #    ("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
             #    ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
             #    ("QDA", QuadraticDiscriminantAnalysis()),\
                 ("GNBayes",GaussianNB()),\
                 ("LDA", LinearDiscriminantAnalysis()), \
                 ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                   ], voting='hard', weights=None)),
             #("VotingSoft", VotingClassifier(estimators=[\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("QDA", QuadraticDiscriminantAnalysis()),\
                 #("GNBayes",GaussianNB()),\
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                 #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:500}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
             #        ], voting='soft', weights=None)),
             ] 
    iterations =10
    #model = SVC()
    model = models[0][1]
    DominantCycle, x_train = acPeriodogram(data,bars=9)
    y_train=pd.Series(data=np.where(gainAhead[x_train.index].values>0,1,-1), index=x_train.index)

    dy = np.zeros_like(y_train)
    dX = np.zeros_like(x_train)

    dy = y_train.values
    dX = x_train.values

    sss = StratifiedShuffleSplit(y_train,iterations,test_size=0.1,
                                 random_state=None)
                                 
    cm_sum_is = np.zeros((2,2))
    cm_sum_oos = np.zeros((2,2))
        
    #  For each entry in the set of splits, fit and predict
    for train_index,test_index in sss:
        X_train, X_test = dX[train_index], dX[test_index]
        Y_train, y_test = dy[train_index], dy[test_index] 

    #  fit the model to the in-sample data
        model.fit(X_train, Y_train)

    #  test the in-sample fit    
        y_pred_is = model.predict(X_train)
        cm_is = confusion_matrix(Y_train, y_pred_is)
        cm_sum_is = cm_sum_is + cm_is

    #  test the out-of-sample data
        y_pred_oos = model.predict(X_test)
        cm_oos = confusion_matrix(y_test, y_pred_oos)
        cm_sum_oos = cm_sum_oos + cm_oos

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

    print "\n\nSymbol is ", ticker
    print "Learning algorithm is", model
    print "Dominant Cycle is", DominantCycle
    print "Confusion matrix for %i randomized tests" % iterations
    print "for years ", data.index[0] , " through ", data.index[-2]  
    print "\nIn sample"
    print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpIS, fnIS, recallIS)
    print "neg:  %i  %i" % (fpIS, tnIS)
    print "      %.2f          %.2f " % (precisionIS, accuracyIS)
    print "f1:   %.2f" % f1IS

    print "\nOut of sample"
    print "     predicted"
    print "      pos neg"
    print "pos:  %i  %i  %.2f" % (tpOOS, fnOOS, recallOOS)
    print "neg:  %i  %i" % (fpOOS, tnOOS)
    print "      %.2f          %.2f " % (precisionOOS, accuracyOOS)
    print "f1:   %.2f" % f1OOS

    print "\nend of run"
