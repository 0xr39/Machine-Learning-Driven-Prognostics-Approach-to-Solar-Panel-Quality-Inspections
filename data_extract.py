#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:51:59 2021

@author: alex
"""
#%%
#Import from sql server
import pandas as pd
from pandas import DataFrame
import pymysql as sql

connection = sql.connect(
    host='cityu.czewygbw4hog.ap-east-1.rds.amazonaws.com',
    user='admin',
    passwd="korcus-fEzhe6-zetbyc",
    db='c_modules',
    port=3306
    )


cursor = connection.cursor()

#%%
#custom sql queries

queries={

    #productName, manuAdd, insDate, orderQuant, evaStandard
    "report" : "SELECT * FROM report",
    
    #cellTech
    "cell_tech" : "SELECT * FROM cell_tech",
    
    #stdName for manu
    "manu" : "SELECT * FROM manufacturer",
    
    #qaService
    "qa" : "SELECT * FROM report__qaservice A LEFT OUTER JOIN qaservice B ON A.`qaservice_id` = B.`qaservice_id`",
    
    #stdName for client
    "client" : "SELECT * FROM client",
    
    #aql, need to turn acceptance into score and encode test name
    "aql" : "SELECT * FROM aql",
    
    #refSampleProvider
    "refSampleProvider" : "SELECT * FROM ref_samples",
    
    #performedTest, use one-hot for all the test
    "performedTest" : "SELECT * FROM report__performed_test A LEFT OUTER JOIN performed_test B ON A.`test_id` = B.`test_id`",
    
    #consConf
    "consConf" : "SELECT * FROM report__product_cons_conf A LEFT OUTER JOIN product_cons_conf B ON A.`pcc_id` = B.`pcc_id`",
    
    #visIns
    "visIns" : "SELECT * FROM `report__visual_inspection` A LEFT OUTER JOIN visual_inspection B ON A.`vi_id` = B.`vi_id`",
    
    #elIns
    "elIns" : "SELECT * FROM `report__el_inspection` A LEFT OUTER JOIN el_inspection B ON A.`el_inspection_id` = B.`el_inspection_id`",
    
    #label
    "label" : "SELECT * FROM `report__label_frame_check` A LEFT OUTER JOIN label_frame_check B ON A.lf_check_id = B.lf_check_id",
    
    #cert
    "cert" : "SELECT * FROM report__mod_fact_cert A LEFT OUTER JOIN mod_fact_cert B ON A.mod_fact_cert_id = B.certification_id",
    
    #inspector
    "inspector" : "SELECT * FROM report__sino_hr A LEFT OUTER JOIN sino_hr B ON A.sino_hr_id = B.sino_hr_id",

}


#%%

def main():
    tableLinks={}
    for i in queries:
        print("loaded queries: ",i)
        cursor.execute(queries[i])
        table = DataFrame(cursor.fetchall())
        jsonPath = "./extracted_queries/"+i+".json"
        print("jsonPath: ",jsonPath)
        tableLinks[i]=jsonPath

    print("finished loading queries")
    return tableLinks