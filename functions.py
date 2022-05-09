
# Import base libraries
import pandas as pd
import numpy as np
from scipy.io import arff

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def ROC_curve_train_test(dataNumber, X_tr, y_tr, X_te, y_te, model, model_name, save=0):
    
    """
    This is a function to draw an ROC curve overlaying training and testing results for selected parameters.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    model: Classifier model, previously fit on training data
    model_name: Title of model
    save: Bool parameter to control saving the plot
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))

    #model = XGBClassifier(**xgbParams)
    #model.fit(X_tr, y_tr)
      
    y_train_pred = model.predict(X_tr)   
    y_train_prob = model.predict_proba(X_tr) #Probability estimates for each class
    fpr_train, tpr_train, thresholds_train = roc_curve(y_tr, y_train_prob[:,1])
    auc_train = round(auc(fpr_train, tpr_train),3)
    f1_train = round(f1_score(y_tr, y_train_pred),3)
    recall_train = round(recall_score(y_tr, y_train_pred),3)
    precision_train = round(precision_score(y_tr, y_train_pred),3)
    accuracy_train = round(accuracy_score(y_tr, y_train_pred),3)
    ax.plot(fpr_train, tpr_train, lw=2, label=f'Train: acc={accuracy_train}, prec={precision_train}, rec={recall_train}, f1={f1_train}, AUC={auc_train}')
    #ax.plot(fpr_train, tpr_train, lw=2, label=f'Train: AUC={auc_train}')
    
    y_test_pred = model.predict(X_te)
    y_test_prob = model.predict_proba(X_te) #Probability estimates for each class
    fpr_test, tpr_test, thresholds_test = roc_curve(y_te, y_test_prob[:,1])
    auc_test = round(auc(fpr_test, tpr_test),3)
    f1_test = round(f1_score(y_te, y_test_pred),3)
    recall_test = round(recall_score(y_te, y_test_pred),3)
    precision_test = round(precision_score(y_te, y_test_pred),3)
    accuracy_test = round(accuracy_score(y_te, y_test_pred),3)
    ax.plot(fpr_test, tpr_test, lw=2, label=f'Test: acc={accuracy_test}, prec={precision_test}, rec={recall_test}, f1={f1_test}, AUC={auc_test}')
    #ax.plot(fpr_test, tpr_test, lw=2, label=f'Test: AUC={auc_test}')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    #ax.set_yticks([i/20.0 for i in range(21)])
    #ax.set_xticks([i/20.0 for i in range(21)])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.set_title(f'ROC Curve for Data {dataNumber}, {model_name}', fontsize=14)
    ax.legend(loc='auto', fontsize=13)
    
    
    if save:
        plt.savefig(f'figures/ROC_Curve_d{dataNumber}_{model_name}.png')


def scan_xgb_overlay(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, scanParam, scanList, save=0):

    """
    This is a function to scan over XGBClassifier parameters. 
    It creates one figures /ROC curve overlaying training and testing results for all scanned values.
    Returns to a table of listing evaluation metrics.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    xgbParams: XGBClassifier parameters used to create the model
    scanParam: the XGBClassifier parameter, which will be scanned
    scanList: the list of the values to be scanned; any size is OK.
    save: Bool parameter to control saving the plot
    """
    
    fig, ax = plt.subplots(1, figsize=(10, 8))
    
    #cp = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    cp = sns.color_palette()
    
    model_scores_list = []
    
    for i,s in enumerate(scanList):
        
        xgbParams[scanParam]=s
        
        clf = XGBClassifier(**xgbParams)
          
        clf.fit(X_tr, y_tr)
    
        y_train_pred = clf.predict(X_tr)   
        y_train_prob = clf.predict_proba(X_tr) #Probability estimates for each class
        fpr_train, tpr_train, thresholds_train = roc_curve(y_tr, y_train_prob[:,1])
        auc_train = round(auc(fpr_train, tpr_train),3)
        f1_train = round(f1_score(y_tr, y_train_pred),3)
        recall_train = round(recall_score(y_tr, y_train_pred),3)
        precision_train = round(precision_score(y_tr, y_train_pred),3)
        accuracy_train = round(accuracy_score(y_tr, y_train_pred),3)
        
        y_test_pred = clf.predict(X_te)
        y_test_prob = clf.predict_proba(X_te) #Probability estimates for each class
        fpr_test, tpr_test, thresholds_test = roc_curve(y_te, y_test_prob[:,1])
        auc_test = round(auc(fpr_test, tpr_test),3)
        f1_test = round(f1_score(y_te, y_test_pred),3)
        recall_test = round(recall_score(y_te, y_test_pred),3)
        precision_test = round(precision_score(y_te, y_test_pred),3)
        accuracy_test = round(accuracy_score(y_te, y_test_pred),3)
       
        prec_diff = precision_train - precision_test
        prec_diff_scaled = prec_diff/precision_test
        
        fit_scores_train = {'Params': f'{scanParam}={s}  Train ',
                        'accuracy': accuracy_train,
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'auc': auc_train,
                        'prec_diff': prec_diff,
                        'prec_diff_scaled': prec_diff_scaled,
                       }
    
        fit_scores_test = {'Params': f'Test',
                        'accuracy': accuracy_test,
                        'precision': precision_test,
                        'recall': recall_test,
                        'f1': f1_test,
                        'auc': auc_test,
                        'prec_diff': prec_diff,
                        'prec_diff_scaled': prec_diff_scaled,
                       }
    
        model_scores_list.append(fit_scores_train)
        model_scores_list.append(fit_scores_test)
    
        ax.plot(fpr_train, tpr_train, lw=2, color=cp[i], linestyle='dashed', label=f"Train, {scanParam}={s}")
        ax.plot(fpr_test, tpr_test, lw=2, color=cp[i], label=f"Test, {scanParam}={s}")  
        

    #ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f"ROC Curve for Data {dataNumber}, Scan '{scanParam}' ", fontsize=14)
    ax.legend(loc='auto', fontsize=13)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.5, 1.05])
        
    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_df = model_scores_df.set_index('Params')
    #print(model_scores_df)
    
    if save:
        plt.savefig(f'figures/ROC_Curve_d{dataNumber}_{scanParam}_overlay.png')
    
    return model_scores_df



def scan_xgb(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, scanParam, scanList, save=0):

    """
    This is a function to scan over XGBClassifier parameters. 
    It creates 6 figures /ROC curve overlaying training and testing results. One figure for each scan value.
    Returns to a table of listing evaluation metrics.
    
    dataNumber: Data file in use
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    xgbParams: XGBClassifier parameters used to create the model
    scanParam: the XGBClassifier parameter, which will be scannes
    scanList: the list of the values to be scanned; The required list size is 6.
    save: Bool parameter to control saving the plot
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    #plt.tight_layout(pad=5)
    
    model_scores_list = []
    
    for ax, s in zip(axes.flat, scanList):
        
        xgbParams[scanParam]=s
        
        clf = XGBClassifier(**xgbParams)
          
        clf.fit(X_tr, y_tr)
    
        y_train_pred = clf.predict(X_tr)   
        y_train_prob = clf.predict_proba(X_tr) #Probability estimates for each class
        fpr_train, tpr_train, thresholds_train = roc_curve(y_tr, y_train_prob[:,1])
        auc_train = round(auc(fpr_train, tpr_train),3)
        f1_train = round(f1_score(y_tr, y_train_pred),3)
        recall_train = round(recall_score(y_tr, y_train_pred),3)
        precision_train = round(precision_score(y_tr, y_train_pred),3)
        accuracy_train = round(accuracy_score(y_tr, y_train_pred),3)
        
        y_test_pred = clf.predict(X_te)
        y_test_prob = clf.predict_proba(X_te) #Probability estimates for each class
        fpr_test, tpr_test, thresholds_test = roc_curve(y_te, y_test_prob[:,1])
        auc_test = round(auc(fpr_test, tpr_test),3)
        f1_test = round(f1_score(y_te, y_test_pred),3)
        recall_test = round(recall_score(y_te, y_test_pred),3)
        precision_test = round(precision_score(y_te, y_test_pred),3)
        accuracy_test = round(accuracy_score(y_te, y_test_pred),3)
       
        fit_scores_train = {'Params': f'{scanParam}={s}  Train ',
                        'accuracy': accuracy_train,
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'auc': auc_train
                       }
    
        fit_scores_test = {'Params': f'Test',
                        'accuracy': accuracy_test,
                        'precision': precision_test,
                        'recall': recall_test,
                        'f1': f1_test,
                        'auc': auc_test
                       }
    
        model_scores_list.append(fit_scores_train)
        model_scores_list.append(fit_scores_test)

    
        ax.plot(fpr_train, tpr_train, lw=2, label=f'Train: acc={accuracy_train}, prec={precision_train}, rec={recall_train}, f1={f1_train}, AUC={auc_train}')
        ax.plot(fpr_test, tpr_test, lw=2, label=f'Test: acc={accuracy_test}, prec={precision_test}, rec={recall_test}, f1={f1_test}, AUC={auc_test}')  
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_yticks([i/20.0 for i in range(21)])
        ax.set_xticks([i/20.0 for i in range(21)])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f'ROC Curve for Data {dataNumber}, {scanParam}={s}', fontsize=14)
        ax.legend(loc='auto', fontsize=13)

    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_df = model_scores_df.set_index('Params')
    #print(model_scores_df)
    
    if save:
        plt.savefig(f'figures/ROC_Curve_d{dataNumber}_{scanParam}.png')
    
    return model_scores_df
    
    