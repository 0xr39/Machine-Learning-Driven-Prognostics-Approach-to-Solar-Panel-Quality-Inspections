#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:24:35 2021

@author: alex
"""
import pandas as pd
import math
#from dateutil import parser #can use later
import datetime

class data_structurer:
    
    def __init__(self, sqlTableLinks):
        self.sqlTableLinks = sqlTableLinks
        
    #write df to excel
    def viewInExcel(self, df, fileName, sheetName):
        from pandas import ExcelWriter
        with ExcelWriter(fileName) as writer:
            df.to_excel(writer, sheet_name=sheetName)
        print("file written")
    
    def readData(self):
        pdDict = {}
        for i in self.sqlTableLinks:
            temp = pd.read_json(self.sqlTableLinks[i])
            pdDict[i]=temp
        return pdDict
    
    def testStructure(self, test):
        resultArray = pd.unique(test[2])
        resultArray = [x for x in resultArray if str(x) != 'None']
        columnData = { "tempReportId": pd.unique(test[0])}
        newTest = pd.DataFrame(data = columnData)
        for i in resultArray:
            newTest[i]= 0
            
        for i in pd.unique(test[0]):
            for index, rows in test.loc[test[0]==i].iterrows():
                series = rows
                result = series.loc[2]
                if series.loc[2] != None:
                    for j in resultArray:
                        if result == j:
                            newTest[j][newTest['tempReportId']==i]=newTest[j][newTest['tempReportId']==i]+1
                
        return newTest
    

        

    def semiAql(self, aql):
        idArray = pd.unique(aql[0])
        columnData = { "tempReportId": idArray}
        newAql = pd.DataFrame(data = columnData)
        newAql["AQL calculated score"] = 0
        critAcc = aql.groupby(0)[2].mean()
        critRej = aql.groupby(0)[3].mean()
        majAcc = aql.groupby(0)[4].mean()
        majRej = aql.groupby(0)[5].mean()
        minAcc = aql.groupby(0)[6].mean()
        minRej = aql.groupby(0)[7].mean()
        for i in idArray:
            value = critAcc[i]*3+critRej[i]*-3+majAcc[i]*2+majRej[i]*-2+minAcc[i]*1+minRej[i]*-1
            newAql["AQL calculated score"][newAql["tempReportId"]==i] = value
        
        return newAql
        
    def addFinder(self, motherData):

        addData = pd.read_excel("manuAdd.xlsx",  engine="openpyxl")

        # if type(add) is str:
        #     add = add.lower()
        #     if add.find("china")>=0 or add.find("hefei")>=0 or add.find("dongtai")>=0 or add.find("yiwu")>=0 or add.find("zhejiang")>=0 or add.find("henan")>=0 or add.find("jiangsu")>=0 or add.find("anhui")>=0 or add.find("ningbo")>=0 or add.find("shanghai")>=0 or add.find("hebei")>=0:
        #         add = "china"
        #     elif add.find("singapore")>=0:
        #         add = "singapore"
        #     elif add.find("france")>=0:
        #         add = "france"
        #     elif add.find("germany")>=0:
        #         add = "germany"
        #     elif add.find("netherland")>=0:
        #         add = "netherland"
        #     elif add.find("spain")>=0:
        #         add = "spain"
        #     elif add.find("vietnam")>=0:
        #         add = "vietnam"
        #     elif add.find("emirates")>=0:
        #         add = "uae"
        #     elif add.find("india")>=0:
        #         add = "india"
        #     elif add.find("australia")>=0:
        #         add = "australia"
        #     elif add.find("yemen")>=0:
        #         add = "yemen"
        #     elif add.find("thailand")>=0:
        #         add = "thailand"
        #     elif add.find("malaysia")>=0:
        #         add = "malaysia"
        #     else:
        #         add = None
        # else:
        #     add = None
        for index, row in motherData.iterrows():
            reportId = row.loc["reportId"]
            manuId = row.loc["manuId"]
            pd.isna(reportId)
            if math.isnan(reportId)==False and math.isnan(manuId)==False:
                resultData = addData["Countries"][addData["manu_id"]==manuId].iloc[0]
                motherData["manuAdd"][motherData["reportId"]==reportId] = resultData


        return motherData
    
    
    def formMaster(self):
        pdDict = self.readData()
        #get&name master
        master = pdDict["report"]
        master = master.rename(columns={0:"reportId", 1:"reportName",2:"reportDate",3:"productName",4:"manuId",5:"nonStdName",6:"manuAdd",7:"insDate",8:"orderQuant",9:"evaStandard",10:"finalResult",11:"clientId",12:"clientNonStdName",13:"refId",14:"remarks"})
        
        #deal with manu
        manu = pdDict["manu"]
        manu = manu.rename(columns={0:"manu_manuId", 1:"manuStdName"})
        master = master.merge(manu, how="left", left_on="manuId", right_on="manu_manuId")
        master = master.drop(columns=["nonStdName", "manu_manuId"])
        #manu done
        
        #manuAdd
        master = self.addFinder(master)
        #manuAdd done, 68 null
        
        #insDate, change to reportDate (8 Nov meeting)
        master = master.drop(columns=["insDate"])
        #reportDate done
        
        #orderQuant hard to structure, dropping for now
        master = master.drop(columns=["orderQuant"])
        #orderQuant done
        
        #qa
        qa = pdDict["qa"]
        qa[3] = qa[3].replace(["FPSI","PSSI","SPSI"],"FPSI/PSSI/SPSI")
        qa[3] = qa[3].replace(["CPM","DuPro"],"CPM/DuPro")
        qa = qa.drop(columns=[1,2])
        qa = qa.rename(columns={0: "qa_reportId", 3: "qaService"})
        dummyQa = pd.get_dummies(qa)
        groupedQa = dummyQa.groupby("qa_reportId").max()#ordered already

        master = master.merge(groupedQa, how="left", left_on="reportId", right_on="qa_reportId")
        #self.viewInExcel(groupedQa, "qaTemp.xlsx", "main")
        #qa done, 30 null
        
        #cell tech
        cellTech = pdDict["cell_tech"]
        cellTech = cellTech.rename(columns={0: "cellTech_reportId", 1: "cellTech"})
        dummyCellTech = pd.get_dummies(cellTech)
        dummyCellTech = dummyCellTech.drop(columns=["cellTech_"])
        groupedCellTech = dummyCellTech.groupby("cellTech_reportId").max()
        
        master = master.merge(groupedCellTech, how="left", left_on="reportId", right_on="cellTech_reportId")
        #self.viewInExcel(dummyCellTech, "cellTech.xlsx", "main")
        
        #evaStandard, 92 null, drop for now
        master = master.drop(columns=["evaStandard"])
        #evaStandard done
        
        #remark, 84 null, a complicated str, drop for now
        master = master.drop(columns=["remarks"])
        #remark done
        
        #clientName
        client = pdDict["client"]
        client = client.rename(columns={0:"client_clientId", 1:"clientStdName"})
        master = master.merge(client, how="left", left_on="clientId", right_on="client_clientId")
        master = master.drop(columns=["clientNonStdName", "clientId", "client_clientId"])
        #client Done
        
        #aql
        # aql = pdDict["aql"]
        # newAql = self.semiAql(aql)
        # master = master.merge(newAql, how="left", left_on="reportId", right_on="tempReportId")
        # master = master.drop(columns=["tempReportId"])
        #aql Done
        
        #ref sample, a lot of null (66)
        master = master.drop(columns=["refId"])
        #ref sample Done
        
        #inspector
        ins = pdDict["inspector"]
        ins = ins.rename(columns={0:"ins_reportId", 1:"ins_id", 3:"ins_name"})
        ins_name = ins.drop(columns=[2])
        #master = master.merge(ins_name, how="left", left_on="reportId", right_on="ins_reportId")
        #for index, row in ins_name.iterrows():
        #    insId = row.loc["ins_id"]
        #    if row.loc["ins_id"] not in [8,5,12,15,17,1,11]:
        #        ins_name["ins_name"][ins_name["ins_id"]==insId] = "other"
        ins_name = ins_name.drop(columns=["ins_id"])
        ins_name = pd.get_dummies(ins_name)
        ins_name = ins_name.groupby("ins_reportId").sum()
        #master = master.drop(columns=["ins_reportId", "ins_id"])
        #groupedIns = ins.groupby(0)["ins_id"].apply(list) #ordered already
        master = master.merge(ins_name, how="left", left_on="reportId", right_on="ins_reportId")
        #master = master.rename(columns={"temp":"inspector"})
        #self.viewInExcel(ins_name, "inspectorRaw.xlsx", "main")
        
        #tests
        
        #elIns
        elIns = pdDict["elIns"]
        elIns[2]=elIns[2].replace("None", "pass")
        newElIns = self.testStructure(elIns)
        master = master.merge(newElIns, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"['Major', 'Minor']":"EL maj min","Major":"EL maj", "Minor":"EL min", 'pass':'EL pass'})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = ((row.loc["EL maj min"])*2)+(row.loc["EL maj"])+(row.loc["EL min"])+(row.loc["EL pass"])
            if countSum == 0:
                countSum+=1
            master["EL maj"][master["reportId"]==reportId] = ((row.loc["EL maj"])+(row.loc["EL maj min"]))/countSum
            master["EL min"][master["reportId"]==reportId] = ((row.loc["EL min"])+(row.loc["EL maj min"]))/countSum
            master["EL pass"][master["reportId"]==reportId] = (row.loc["EL pass"])/countSum
        master = master.drop(columns=["EL maj min"])
        #elIns Done
        
        #VI
        vi = pdDict["visIns"]
        vi[2]=vi[2].replace("None", "pass")
        vi1 = vi.loc[vi[4]==1]
        vi2 = vi.loc[vi[4]==2]
        
        newVi1 = self.testStructure(vi1)
        newVi2 = self.testStructure(vi2)
        #vi1
        master = master.merge(newVi1, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"['Major', 'Minor']":"VI1 maj min","Major":"VI1 maj", "Minor":"VI1 min", "pass":"VI1 pass"})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = ((row.loc["VI1 maj min"])*2)+(row.loc["VI1 maj"])+(row.loc["VI1 min"])+(row.loc["VI1 pass"])
            if countSum == 0:
                countSum+=1
            master["VI1 maj"][master["reportId"]==reportId] = ((row.loc["VI1 maj"])+(row.loc["VI1 maj min"]))/countSum
            master["VI1 min"][master["reportId"]==reportId] = ((row.loc["VI1 min"])+(row.loc["VI1 maj min"]))/countSum
            master["VI1 pass"][master["reportId"]==reportId] = (row.loc["VI1 pass"])/countSum
        master = master.drop(columns=["VI1 maj min"])
        
        #vi2
        master = master.merge(newVi2, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"['Major', 'Minor']":"VI2 maj min","Major":"VI2 maj", "Minor":"VI2 min", "pass":"VI2 pass"})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = ((row.loc["VI2 maj min"])*2)+(row.loc["VI2 maj"])+(row.loc["VI2 min"])+(row.loc["VI2 pass"])
            if countSum == 0:
                countSum+=1
            master["VI2 maj"][master["reportId"]==reportId] = ((row.loc["VI2 maj"])+(row.loc["VI2 maj min"]))/countSum
            master["VI2 min"][master["reportId"]==reportId] = ((row.loc["VI2 min"])+(row.loc["VI2 maj min"]))/countSum
            master["VI2 pass"][master["reportId"]==reportId] = (row.loc["VI2 pass"])/countSum
        master = master.drop(columns=["VI2 maj min"])
        
        #VI Done
        
        #consConf
        cons = pdDict["consConf"]
        newCons = self.testStructure(cons)
        master = master.merge(newCons, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"Approved":"Cons Approve","Not applicable":"Cons NA"})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = (row.loc["Cons Approve"])+(row.loc["Cons NA"])
            if countSum == 0:
                countSum+=1
            master["Cons Approve"][master["reportId"]==reportId] = (row.loc["Cons Approve"])/countSum
            master["Cons NA"][master["reportId"]==reportId] = (row.loc["Cons NA"])/countSum
        #master = master.drop(columns=["Cons NA"])
        #consConf Done
        
        #performed Test
        perfTest = pdDict["performedTest"]
        newPerfTest = self.testStructure(perfTest)
        master = master.merge(newPerfTest, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"Conformed":"PT Conformed","Not conformed":"PT !Conformed", "Not applicable":"PT NA", "Partly conformed":"PT 0.5Conformed"})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = (row.loc["PT Conformed"])+(row.loc["PT !Conformed"])+(row.loc["PT NA"])+(row.loc["PT 0.5Conformed"])
            if countSum == 0:
                countSum+=1
            master["PT Conformed"][master["reportId"]==reportId] = (row.loc["PT Conformed"])/countSum
            master["PT !Conformed"][master["reportId"]==reportId] = (row.loc["PT !Conformed"])/countSum
            master["PT NA"][master["reportId"]==reportId] = (row.loc["PT NA"])/countSum
            master["PT 0.5Conformed"][master["reportId"]==reportId] = (row.loc["PT 0.5Conformed"])/countSum
        #PT Done
        
        #label
        label = pdDict["label"]
        newLabel = self.testStructure(label)
        master = master.merge(newLabel, how="left", left_on="reportId", right_on="tempReportId")
        master = master.drop(columns=["tempReportId"])
        master = master.rename(columns={"Conformed":"Label Conformed","Not conformed":"Label !Conformed"})
        for index, row in master.iterrows():
            reportId = row.loc["reportId"]
            countSum = (row.loc["Label Conformed"])+(row.loc["Label !Conformed"])
            if countSum == 0:
                countSum+=1
            master["Label Conformed"][master["reportId"]==reportId] = (row.loc["Label Conformed"])/countSum
            master["Label !Conformed"][master["reportId"]==reportId] = (row.loc["Label !Conformed"])/countSum
        #master = master.drop(columns=["Label !Conformed"])
        #label Done
        
        #cert
        cert = pdDict["cert"]
        cert1 = cert.loc[cert[4]==1]
        cert2 = cert.loc[cert[4]==2]
        master["cert1"]=0
        master["cert2"]=0
        idList1 = pd.unique(cert1[0])
        idList2 = pd.unique(cert2[0])
        groupedCert1 = cert1.groupby(0)[4].count()
        groupedCert2 = cert2.groupby(0)[4].count()
        for i in idList1:
            master["cert1"][master["reportId"]==i]=groupedCert1[i]
        for i in idList2:
            master["cert2"][master["reportId"]==i]=groupedCert2[i]
        #cert2 all null
        master=master.drop(columns=["cert2"])
        #self.viewInExcel(cert, "certTemp.xlsx", "main")
        
        #Final result standardise
        for index, row in master.iterrows():
            target = None
            reportId = row.loc["reportId"]
            target = row.loc["finalResult"]
            if target != None:

                target = target.lower()
                
                if target.find("fail")>=0:
                    master["finalResult"][master["reportId"]==reportId]="!pass"
                elif target.find("with")>=0:
                    master["finalResult"][master["reportId"]==reportId]="!pass"
                elif target.find("not")>=0:
                    master["finalResult"][master["reportId"]==reportId]="!pass"
                elif target.find("inconclusive")>=0:
                    master["finalResult"][master["reportId"]==reportId]="!pass"
                elif target.find("pass")>=0:
                    master["finalResult"][master["reportId"]==reportId]="pass"
                else:
                    master["finalResult"][master["reportId"]==reportId]="!pass"
            

        #finalize feature space
        master = master.drop(columns=["reportName", "reportId", "manuId"])
        
        return master