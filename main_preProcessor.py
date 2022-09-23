#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:12:23 2021

@author: alex
"""
#%%
#data extract and structuring
import data_extract
from data_structurer import data_structurer
import pandas as pd
import numpy as np



sqlTableLinks = data_extract.main()


structurer = data_structurer(sqlTableLinks)

pdDict = structurer.readData()

master = structurer.formMaster()

#%%

#write df to excel
def viewInExcel(df, fileName, sheetName):
    from pandas import ExcelWriter
    with ExcelWriter(fileName) as writer:
        df.to_excel(writer, sheet_name=sheetName)
    print("file written")
    
#write to excel
viewInExcel(master, "motherData.xlsx", "main")

#%%
#info

#get null of each feature
nullDf = master.isnull().sum(axis = 0)
nullDf = nullDf.astype('float64')
for index,value in nullDf.iteritems():
    nullDf[index]= value/(master.shape[0])
viewInExcel(nullDf, "null_percentage.xlsx", "main")

#get cols
colsDf = pd.DataFrame(data={"cols":master.columns, "dtypes": master.dtypes})

#class distribution
classSeries = master.groupby("finalResult")["finalResult"].count()
