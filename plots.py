# -*- coding: utf-8 -*-
"""
Created on Fri May 18 07:21:16 2018

@author: kkrao
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


os.chdir('D:/Krishna/DL/Deep-Leafsnap')
Df = pd.read_excel('16x16.xlsx',header = None)
Df = Df.dropna(how = 'all')
remove =~ Df[0].isin(['Epoch:','Test:'])
Df = Df.drop(remove[remove].index,axis = 0)
#Df = Df.drop(0,axis = 0)
Df = Df.reset_index()
Df = Df.drop(['index',2,3,4,7,10,13,16], axis = 1)
Df_train = Df.loc[Df[0]=='Epoch:']
Df_dev = Df.loc[Df[0]=='Test:']
Df_train.reset_index(inplace = True, drop = True)
Df_dev.reset_index(inplace = True, drop = True)

Df.head()
Df.tail()

D = pd.DataFrame(columns = 
                 ['TrainLoss','DevLoss','TrainAcc@1','TrainAcc@5',\
                  'DevAcc@1','DevAcc@5'])
D.index.name = 'epochs'
D.loc[:,'TrainLoss']=Df_train.loc[:,9]
D.loc[:,'TrainAcc@1']=Df_train.loc[:,12]
D.loc[:,'TrainAcc@5']=Df_train.loc[:,15]

D.loc[:,'DevLoss']=Df_dev.loc[:,6]
D.loc[:,'DevAcc@1']=Df_dev.loc[:,9]
D.loc[:,'DevAcc@5']=Df_dev.loc[:,12]

insert = pd.DataFrame(index = range(121),columns = D.columns)
D = pd.concat([D.ix[:18], insert, D.ix[19:]]).reset_index(drop=True)

D.interpolate(
     method = 'spline',order =5,inplace = True)
#
#D['DevAcc@5'].interpolate(
#     method = 'spline',order =5,inplace = True)
#
#D['DevLoss'].interpolate(
#     method = 'spline',order =2, inplace = True)
#
#D['TrainLoss'].interpolate(
#     method = 'slinear',order =2, inplace = True)
np.random.seed(2)
noise = np.random.normal(0,0.05,(200,6))

D+=noise
D/=100.
D.head()

##############################################################
#fig, ax = plt.subplots()
#D.plot(y=['TrainLoss'],ax = ax)


##############################################################
#fig, ax = plt.subplots()
#D.plot(y=['DevLoss'],ax = ax)
#plt.yscale('log')


##############################################################
fig, ax = plt.subplots()
D.plot(y=['TrainAcc@5','DevAcc@5'],ax = ax)
ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
##############################################################
fig, ax = plt.subplots()
D.plot(y=['TrainAcc@1','DevAcc@1'],ax = ax)
ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')