import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import miceforest as mf

#write df to excel
def viewInExcel(df, fileName, sheetName):
    from pandas import ExcelWriter
    with ExcelWriter(fileName) as writer:
        df.to_excel(writer, sheet_name=sheetName)
    print("file written")

# Read CSV file
data = pd.read_excel("motherData.xlsx", engine="openpyxl").drop(columns=['Unnamed: 0'])

# Drop nan in Target
data = data.dropna(axis=0, subset=['finalResult'])

# Check Missing Value 

# count the num of missing values
missing=data.isnull().sum().reset_index().rename(columns={0:'missNum'})
# calculate proportion
missing['missRate']=missing['missNum']/data.shape[0]
# sort the attributes by missing rate
miss_analy=missing[missing.missRate>0].sort_values(by='missRate',ascending=False)


# Using MICE imputation to handle missing value

#convert object column to category, required by miceforest


# Create kernel. 
kernel = mf.MultipleImputedKernel(
  data,
  save_all_iterations=True,
  datasets = 5
)


kernel.mice(iterations=5, boosting='gbdt', min_sum_hessian_in_leaf=0.01)


# Return the completed kernel data
completed_dataset = kernel.complete_data(dataset=0)
viewInExcel(completed_dataset, "imputed_dataset.xlsx", "main")

#%%

# Dummy
def myDummy(X):
    X=pd.get_dummies(X,columns =['productName','manuAdd', "manuStdName", "clientStdName"])
    return X

# Label
def myLabel(X):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    X['productName'] = labelencoder.fit_transform(X['productName'])
    X['manuAdd'] = labelencoder.fit_transform(X['manuAdd'])
    X['manuStdName'] = labelencoder.fit_transform(X['manuStdName'])
    X['clientStdName'] = labelencoder.fit_transform(X['clientStdName'])
    return X
    





# Handle imbalance data

# ADASYN
def myADA(X_train, X_test, y_train, y_test):
    from imblearn.over_sampling import ADASYN
    
    oversample = ADASYN()
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    X_balanced, y_balanced = oversample.fit_resample(X_train, y_train)
    #X_test_balanced, y_test_balanced = oversample.fit_resample(X_test, y_test)
    df_X_balanced = pd.DataFrame(X_balanced)
    df_y_balanced = pd.DataFrame(y_balanced)
    #viewInExcel(df_y_balanced, "training_y.xlsx", "main")
    return X_balanced, y_balanced, X_test, y_test




# Model training
def trainXgbParams(paramRange, X_balanced, y_balanced):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    
    #train gscv
    xgb = XGBClassifier(verbosity=0)
    clf = GridSearchCV(estimator=xgb, 
                   param_grid=paramRange,
                   scoring='accuracy', 
                   verbose=0,
                   cv=6,
                   n_jobs=-1)

    clf = clf.fit(X_balanced, y_balanced)
    
    #train xgb with best param
    
    bestParams = clf.best_params_
    
    tunedModel = XGBClassifier(**bestParams)
    tunedModel = tunedModel.fit(X_balanced, y_balanced)
    
    untunedModel = XGBClassifier(verbosity=0)
    untunedModel = untunedModel.fit(X_balanced, y_balanced)
    
    tunedScores = cross_val_score(tunedModel, X_balanced, y_balanced, cv=6,scoring='accuracy')
    tunedAcc = tunedScores.mean()
    
    untunedScores = cross_val_score(untunedModel, X_balanced, y_balanced, cv=6,scoring='accuracy')
    untunedAcc = untunedScores.mean()
    
    print("model: ",tunedModel.get_params())
    
    print("Tuned Model Accuracy: ", tunedAcc)
    print("Untuned Model Accuracy: ", untunedAcc)
    print("Best parameters:", bestParams)

    
    return bestParams

def myXgboost(X_balanced, y_balanced, X_test_balanced, y_test_balanced, X, threshold, bestParams):
    from xgboost import XGBClassifier
    
    #make predictions
    xgbModel = XGBClassifier(**bestParams)
    xgbModel = xgbModel.fit(X_balanced, y_balanced)

    y_pre = xgbModel.predict_proba(X_test_balanced)
    y_pre = pd.DataFrame(y_pre)

    for index, row in y_pre.iterrows():
        if row.loc[1] > threshold:
            y_pre.iloc[index] = "pass"
        else:
            y_pre.iloc[index] = "!pass"
    y_pre = y_pre.drop(columns=[1])


    #y_pre = xgbModel.predict(X_test_balanced)
    
    
    
    fiDf = pd.DataFrame(columns=["feature","score"])
    for col,score in zip(X.columns,xgbModel.feature_importances_):
        fiDf=fiDf.append({"feature": col, "score": score},ignore_index=True)
    viewInExcel(fiDf, "feature_importance.xlsx", "main")
        
    # Review the report 
    from sklearn.metrics import classification_report
    from sklearn.metrics import log_loss
    
    print("XGBoost: ")
    print(classification_report(y_test_balanced, y_pre))
    
    logLoss = log_loss(y_test_balanced, xgbModel.predict_proba(X_test_balanced))
    print("Log Loss: ", logLoss)
    return xgbModel , y_pre, logLoss

def myDtree(X_balanced, y_balanced, X_test_balanced, y_test_balanced, X, depth):
    from sklearn import tree
    import matplotlib.pyplot as plt
    from dtreeviz.trees import dtreeviz
    from sklearn.preprocessing import LabelEncoder
    
    y_labeled = LabelEncoder().fit_transform(y_balanced)
    dtree = tree.DecisionTreeClassifier(max_depth=depth)
    dtreeModel = dtree.fit(X_balanced, y_labeled)


    viz = dtreeviz(dtree, 
               X_balanced, 
               y_labeled,
               target_name='Pass/!Pass',
               feature_names=X.columns, 
               class_names=["!pass", "pass"]# need class_names for classifier
              )  
    
    # treeVis = tree.export_text(dtreeModel)
    # tree.plot_tree(dtreeModel, filled=True)
    #fig=plt.figure()
    viz.view()
    #fig.savefig("decistion_tree.png")
    
    y_pre = dtreeModel.predict(X_test_balanced)
    
    print(y_pre)
    y_test_labeled = LabelEncoder().fit_transform(y_test_balanced)
    print(y_test_labeled)
    from sklearn.metrics import classification_report
    print("DTree: ")
    print(classification_report(y_test_labeled, y_pre))
    
    return dtreeModel



# plot confusion matrix
def plotCM(y_test, y_pre , title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pre)
    sampleSize = np.sum(cm)
    if (cm[0,0]+cm[1,0]) > 0:
        nPassPre = cm[0,0]/(cm[0,0]+cm[1,0])#precision
    else:
        nPassPre = 1
        
    if (cm[0,1]+cm[1,1]) > 0 :
        yPassPre = cm[1,1]/(cm[0,1]+cm[1,1]) #precision
    else:
        yPassPre = 1
    acc = (cm[0,0]+cm[1,1])/sampleSize
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["!pass", "pass"])
    disp.plot()
    plt.title(title)

    
    return acc, nPassPre, yPassPre

#%% Pipeline & Testing
completed_dataset = pd.read_excel("imputed_dataset.xlsx", engine="openpyxl").drop(columns=['Unnamed: 0'])
without_test_dataset = completed_dataset.drop(columns=['EL pass','EL min', "EL maj", 'VI1 min', 'VI1 pass', 'VI1 maj', 'VI2 pass', 'VI2 maj', 'VI2 min', 'Cons Approve', 'Cons NA', 'PT Conformed', 'PT !Conformed', 'PT NA', 'PT 0.5Conformed', 'Label Conformed', 'Label !Conformed'])

def dummy_ADA_xgb(data, threshold, bestParams):
    X=data.drop(['finalResult'], axis=1)
    Y=data.loc[0:,'finalResult']
    X=myDummy(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.1, stratify=Y)
    
    X_balanced, y_balanced, X_test_balanced, y_test_balanced = myADA(X_train, X_test, y_train, y_test)
    
    tunedModel, y_pre, logLoss = myXgboost(X_balanced, y_balanced, X_test_balanced, y_test_balanced, X, threshold, bestParams)
    
    acc, nPassPre, yPassPre = plotCM(y_test_balanced, y_pre, "dummy_ADA_xgb")

    return tunedModel, acc, nPassPre, yPassPre, logLoss

def getParams(data):
    params = {
        'min_child_weight': [0.1, 0.2, 0.3, 0.5,1 ], #default 1
        'colsample_bytree': [0.01,0.05, 0.2,1],#default 1, ratio
        'max_depth': [2, 4, 6,8], #default 6
        'eta': [0.15,0.3,0.45], #default 0.3
        'subsample': [0.8,0.9,1],
        }
    X=data.drop(['finalResult'], axis=1)
    Y=data.loc[0:,'finalResult']
    X=myDummy(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.1, stratify=Y)
    
    X_balanced, y_balanced, X_test_balanced, y_test_balanced = myADA(X_train, X_test, y_train, y_test)
    
    bestParams = trainXgbParams(params, X_balanced, y_balanced)
    
    return bestParams
    


    
def plotDecisionTree(data, depth):
    X=data.drop(['finalResult'], axis=1)
    Y=data.loc[0:,'finalResult']
    X=myDummy(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.1, stratify=Y)
    
    X_balanced, y_balanced, X_test_balanced, y_test_balanced = myADA(X_train, X_test, y_train, y_test)
    
    model = myDtree(X_balanced, y_balanced, X_test_balanced, y_test_balanced, X, depth)
    
    return model



def getPerformanceDummy(data, threshold, loop, bestParams):
    import matplotlib.pyplot as plt
    import statistics
    accRate = []
    nPre = []
    yPre = []
    logLossArr = []
    for i in range(0,loop):
        model, acc, nPassPre, yPassPre, logLoss = dummy_ADA_xgb(data, threshold, bestParams)
        accRate.append(acc)
        nPre.append(nPassPre)
        yPre.append(yPassPre)
        logLossArr.append(logLoss)
    meanAcc = statistics.mean(accRate)
    sdAcc = statistics.stdev(accRate)
    meanNPre = statistics.mean(nPre)
    sdNPre = statistics.stdev(nPre)
    meanYPre = statistics.mean(yPre)
    sdYPre = statistics.stdev(yPre)
    meanLogLoss = statistics.mean(logLossArr)
    sdLogLoss = statistics.stdev(logLossArr)
    print("-----Overall Report(dummy_ADA_xgb)------")
    print("mean accuracy: ", meanAcc)
    print("accuracy SD: ", sdAcc)
    print("mean Not Pass precision: ", meanNPre)
    print("Not Pass precision SD: ", sdNPre)
    print("mean Pass precision: ", meanYPre)
    print("Pass precision SD: ", sdYPre)
    print("mean Log Loss: ", meanLogLoss)
    print("Log Loss SD: ", sdLogLoss)
    plt.figure(figsize=(10,6))
    plt.hist(accRate, bins = 50)
    plt.title('Accuracy')
    plt.figure(figsize=(10,6))
    plt.hist(nPre, bins = 50)
    plt.title('Not Pass Precision')
    plt.figure(figsize=(10,6))
    plt.hist(yPre, bins = 50)
    plt.title('Pass Precision')
    
    return meanAcc, sdAcc, meanNPre, sdNPre, meanYPre, sdYPre



def bestThreshold(data, loop, bestParams):
    import matplotlib.pyplot as plt
    
    accRate = []
    nPre = []
    yPre = []
    for i in range(0, 105, 5):
        meanAcc, sdAcc, meanNPre, sdNPre, meanYPre, sdYPre = getPerformanceDummy(data, i/100, loop, bestParams)
        accRate.append(meanAcc)
        nPre.append(meanNPre)
        yPre.append(meanYPre)
        
    plt.figure(figsize=(10,6))
    plt.plot(range(0, 105, 5),accRate,color='blue', linestyle='dashed', label ="accuracy")
    # plt.title('Accuracy')
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # plt.savefig("accuracy vs threshold.png")
    

    plt.plot(range(0, 105, 5),nPre,color='red', linestyle='dashed', label = "!Pass Precision")
    # plt.title('Not Pass Precision')
    # plt.xlabel('Threshold')
    # plt.ylabel('Not Pass Precision')
    # plt.savefig("Not Pass Precision vs threshold.png")
    
    plt.plot(range(0, 105, 5),yPre,color='green', linestyle='dashed', label = "Pass Precision")
    # plt.title('Pass Precision')
    # plt.xlabel('Threshold')
    # plt.ylabel('Pass Precision')
    # plt.savefig("Pass Precision vs threshold.png")
    
    plt.title('Performance under different threshold')
    plt.legend()
    plt.show()
    plt.savefig("Threshold.png")

  
#%% 
bestParams = getParams(completed_dataset)
#%%
#getPerformanceDummy(completed_dataset,0.1, 5, bestParams)
#getPerformanceDummy(without_test_dataset,0.15, 20, bestParams)
#%%
#bestThreshold(completed_dataset, 10, bestParams)
#bestThreshold(without_test_dataset, 10, bestParams)
#%%
plotDecisionTree(completed_dataset, 10)
plotDecisionTree(without_test_dataset, 10) #cant use spyder, use terminal
#print("best params: ", bestParams)


