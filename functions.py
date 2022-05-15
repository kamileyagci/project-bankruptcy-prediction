
# Import base libraries
import pandas as pd
import numpy as np
from scipy.io import arff

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.utils import class_weight
from sklearn.metrics import classification_report


def xgb_model_report(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, model_name, weights=0, save_model=0, print_report=0):

    """
    This is a function to that runs XGBClassifier with the given parameters and print the classification report. 
    Returns to the model.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    xgbParams: XGBClassifier parameters used to create the model
    weights: Bool parameter to use sample_weights or not.
    save_model: Bool parameter to control saving the model
    print_report: Bool parameter to control printing the report
    """
    
    weigths_train = None
    
    if weights:
        print('Sample weights are used!\n--------\n')
        weigths_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_tr)
        
    d_eval_set = [(X_tr, y_tr), (X_te, y_te)]  
    
    clf = XGBClassifier(**xgbParams) 
    clf.fit(X_tr, y_tr, sample_weight=weigths_train, eval_set=d_eval_set, verbose=False)
    
    if print_report:
        print(f'Data {dataNumber} Classification Report:\n')
        print('Training Data:\n', classification_report(y_tr, clf.predict(X_tr)))
        print('Testing Data:\n', classification_report(y_te, clf.predict(X_te)))
    
    if save_model:
        clf.save_model(f'saved_model_history/xgb_data{dataNumber}_{model_name}.json')
           
    return clf


def plot_ROC(dataNumber, X_tr, y_tr, X_te, y_te, model, model_name, save=0):
    
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
    
    y_test_pred = model.predict(X_te)
    y_test_prob = model.predict_proba(X_te) #Probability estimates for each class
    fpr_test, tpr_test, thresholds_test = roc_curve(y_te, y_test_prob[:,1])
    auc_test = round(auc(fpr_test, tpr_test),3)
    f1_test = round(f1_score(y_te, y_test_pred),3)
    recall_test = round(recall_score(y_te, y_test_pred),3)
    precision_test = round(precision_score(y_te, y_test_pred),3)
    accuracy_test = round(accuracy_score(y_te, y_test_pred),3)

 
    label_train = f"Train:  prec={precision_train}, rec={recall_train}, f1={f1_train}, acc={accuracy_train}, AUC={auc_train}"
    label_test = f"Test: prec={precision_test}, rec={recall_test}, f1={f1_test}, acc={accuracy_test}, AUC={auc_test}"
    ax.plot(fpr_train, tpr_train, lw=2, linestyle='dashed', label=label_train)
    ax.plot(fpr_test, tpr_test, lw=2, label=label_test) 
    
    
    ax.plot([0, 1], [0, 1], color='0.7', lw=2, linestyle='-.')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    #ax.set_yticks([i/20.0 for i in range(21)])
    #ax.set_xticks([i/20.0 for i in range(21)])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.set_title(f'ROC Curve for Data {dataNumber}, {model_name}', fontsize=14)
    ax.legend(loc='auto', fontsize=14)
    
    if save:
        plt.savefig(f'figures/ROC_data{dataNumber}_{model_name}.png')

        

def plot_logloss(dataNumber, model, model_name, save=0):

    """
    This is a function to plot the Log Loss.
    Returns to None.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    model: Trained model
    model_name: Title of the model
    save: Bool parameter to control saving the plot
    """
    
    logloss_results = model.evals_result()
    epochs = len(logloss_results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x_axis, logloss_results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, logloss_results['validation_1']['logloss'], label='Test')
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Log Loss',fontsize=14)
    ax.set_title(f'Data {dataNumber}, Log Loss for {model_name}', fontsize=14)
    ax.legend(loc='auto', fontsize=14)
    
    if save:
        plt.savefig(f'figures/LogLoss_data{dataNumber}_{model_name}.png')

        
                
def scan_xgb_ROC_metrics(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, scanParam, scanList, weights=0, plot=1, save=0):

    """
    This is a function to scan over XGBClassifier parameters. 
    It creates two figures: 
    1) ROC curve overlaying training and testing results for all scanned values
    2) Evaluation metrics overlaying training and testing results for all scanned values.
    Returns to a dataframe of listing evaluation metrics.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    xgbParams: XGBClassifier parameters used to create the model
    scanParam: the XGBClassifier parameter, which will be scanned
    scanList: the list of the values to be scanned; any size is OK.
    weights: Bool parameter to use sample_weights or not.
    plot: Bool parameter to control creating the plot
    save: Bool parameter to control saving the plot
    """
    
    if plot:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        cp = sns.color_palette()
        #cp = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        cp = sns.color_palette()
    
    model_scores_list = []
    
    weigths_train = None
    
    if weights:
        print('Sample weights are used!\n--------\n')
        weigths_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_tr)
    
    for i,s in enumerate(scanList):
        
        xgbParams[scanParam]=s
        
        clf = XGBClassifier(**xgbParams) 
        clf.fit(X_tr, y_tr, sample_weight=weigths_train)
    
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
       
        prec_diff_sc = (precision_train - precision_test)/precision_test
        rec_diff_sc = (recall_train - recall_test)/recall_test
        f1_diff_sc = (f1_train - f1_test)/f1_test
        #overfit_measure = np.sqrt((1/3)*(prec_diff_sc**2 + rec_diff_sc**2 + f1_diff_sc**2))
        overfit_measure = np.mean([prec_diff_sc, rec_diff_sc, f1_diff_sc])
        
        fit_scores_train = {'Params': f'{scanParam}={s}  Train ',
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'accuracy': accuracy_train,
                        'auc': auc_train,
                        'prec_diff_sc': prec_diff_sc,
                        'rec_diff_sc': rec_diff_sc,
                        'f1_diff_sc': f1_diff_sc,
                        'overfit_measure': overfit_measure,
                       }
    
        fit_scores_test = {'Params': f'Test',
                        'precision': precision_test,
                        'recall': recall_test,
                        'f1': f1_test,
                        'accuracy': accuracy_test,
                        'auc': auc_test,
                       }
    
        model_scores_list.append(fit_scores_train)
        model_scores_list.append(fit_scores_test)
    
        if plot:
            ax.plot(fpr_train, tpr_train, lw=2, color=cp[i], linestyle='dashed', label=f"Train, {scanParam}={s}, AUC={auc_train}")
            ax.plot(fpr_test, tpr_test, lw=2, color=cp[i], label=f"Test, {scanParam}={s}, AUC={auc_test}")  


    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_df = model_scores_df.set_index('Params')
    #print(model_scores_df)
            
    if plot:    
        #ax.plot([0, 1], [0, 1], color='0.7', lw=2, linestyle='-.')
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f"ROC Curve for Data {dataNumber}, Scan '{scanParam}' ", fontsize=14)
        ax.legend(loc='auto', fontsize=13)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.5, 1.05])
        
        
    if plot:
        select_row_train = [i*2 for i in range(0, len(scanList))]
        select_row_test = [i*2+1 for i in range(0, len(scanList))]
        rec_train_list = model_scores_df['recall'].iloc[select_row_train]
        rec_test_list = model_scores_df['recall'].iloc[select_row_test]
        f1_train_list = model_scores_df['f1'].iloc[select_row_train]
        f1_test_list = model_scores_df['f1'].iloc[select_row_test]
        prec_train_list = model_scores_df['precision'].iloc[select_row_train]
        prec_test_list = model_scores_df['precision'].iloc[select_row_test]
        acc_train_list = model_scores_df['accuracy'].iloc[select_row_train]
        acc_test_list = model_scores_df['accuracy'].iloc[select_row_test]

        ax2.plot(scanList, rec_train_list, 'ro--', label="Recall Train")
        ax2.plot(scanList, rec_test_list, 'ro-', label="Recall Test")
        ax2.plot(scanList, f1_train_list, 'go--', label="F1-score Train")
        ax2.plot(scanList, f1_test_list, 'go-', label="F1-score Test")
        ax2.plot(scanList, prec_train_list, 'yo--', label="Precision Train")
        ax2.plot(scanList, prec_test_list, 'yo-', label="Precision Test")
        ax2.plot(scanList, acc_train_list, 'bo--', label="Accuracy Train")
        ax2.plot(scanList, acc_test_list, 'bo-', label="Accuracy Test")
        
        ax2.set_xlabel(scanParam, fontsize=14)
        ax2.set_ylabel('Metric Value', fontsize=14)
        ax2.set_title(f"Evaluation Metrics for Data {dataNumber}, Scan '{scanParam}' ", fontsize=14)
        ax2.legend(loc='auto', fontsize=13)
        ax2.set_ylim([0, 1.05])
    
    
    if save and plot:
        plt.savefig(f'figures/ROC_Metrics_d{dataNumber}_{scanParam}.png')

    return model_scores_df



def scan_xgb_logloss_metrics(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, scanParam, scanList, weights=0, plot=1, save=0):

    """
    This is a function to scan over XGBClassifier parameters. 
    It creates two figures: 
    1) Log Loss overlaying training and testing results for all scanned values
    2) Evaluation metrics overlaying training and testing results for all scanned values.
    Returns to a table of listing evaluation metrics.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    xgbParams: XGBClassifier parameters used to create the model
    scanParam: the XGBClassifier parameter, which will be scanned
    scanList: the list of the values to be scanned; any size is OK.
    weights: Bool parameter to use sample_weights or not.
    plot: Bool parameter to control creating the plot
    save: Bool parameter to control saving the plot
    """
    
    if plot:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        cp = sns.color_palette()
    
    model_scores_list = []
    
    weigths_train = None
    
    if weights:
        print('Sample weights are used!\n--------\n')
        weigths_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_tr)
    
    d_eval_set = [(X_tr, y_tr), (X_te, y_te)]
    
    for i,s in enumerate(scanList):
        
        xgbParams[scanParam]=s
        
        clf = XGBClassifier(**xgbParams) 
        clf.fit(X_tr, y_tr, sample_weight=weigths_train, eval_set=d_eval_set, verbose=False)
    
        y_train_pred = clf.predict(X_tr) 
        f1_train = round(f1_score(y_tr, y_train_pred),3)
        recall_train = round(recall_score(y_tr, y_train_pred),3)
        precision_train = round(precision_score(y_tr, y_train_pred),3)
        accuracy_train = round(accuracy_score(y_tr, y_train_pred),3)
        
        y_test_pred = clf.predict(X_te)
        f1_test = round(f1_score(y_te, y_test_pred),3)
        recall_test = round(recall_score(y_te, y_test_pred),3)
        precision_test = round(precision_score(y_te, y_test_pred),3)
        accuracy_test = round(accuracy_score(y_te, y_test_pred),3)
       
        prec_diff_sc = (precision_train - precision_test)/precision_test
        rec_diff_sc = (recall_train - recall_test)/recall_test
        f1_diff_sc = (f1_train - f1_test)/f1_test
        overfit_measure = np.mean([prec_diff_sc, rec_diff_sc, f1_diff_sc])
        
        logloss_results = clf.evals_result()
        logloss_results_train = logloss_results['validation_0']['logloss']
        logloss_results_test = logloss_results['validation_1']['logloss']
        logloss_train = round(logloss_results_train[-1], 3)
        logloss_test = round(logloss_results_test[-1], 3)
        
        fit_scores_train = {'Params': f'{scanParam}={s}  Train ',
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'accuracy': accuracy_train,
                        'logloss': logloss_train
                        #'prec_diff_sc': prec_diff_sc,
                        #'rec_diff_sc': rec_diff_sc,
                        #'f1_diff_sc': f1_diff_sc,
                        #'overfit_measure': overfit_measure,
                       }
    
        fit_scores_test = {'Params': f'Test',
                        'precision': precision_test,
                        'recall': recall_test,
                        'f1': f1_test,
                        'accuracy': accuracy_test,
                        'logloss': logloss_test,
                       }
    
        model_scores_list.append(fit_scores_train)
        model_scores_list.append(fit_scores_test)
    
        if plot:
            # Plot logloss
            if i==0:
                epochs = len(logloss_results_train)
                x_axis = range(0, epochs)
            ax.plot(x_axis, logloss_results_train, lw=2, color=cp[i], linestyle='dashed', label=f"Train, {scanParam}={s}")
            ax.plot(x_axis, logloss_results_test, lw=2, color=cp[i], label=f"Test, {scanParam}={s}")  


    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_df = model_scores_df.set_index('Params')
            
    if plot:
        # logloss plot settings
        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Log Loss', fontsize=14)
        ax.set_title(f'Data {dataNumber}, Log Loss', fontsize=14)
        ax.legend(loc='auto', fontsize=13)
        #ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        
        #Plot metrics
        select_row_train = [i*2 for i in range(0, len(scanList))]
        select_row_test = [i*2+1 for i in range(0, len(scanList))]
        rec_train_list = model_scores_df['recall'].iloc[select_row_train]
        rec_test_list = model_scores_df['recall'].iloc[select_row_test]
        f1_train_list = model_scores_df['f1'].iloc[select_row_train]
        f1_test_list = model_scores_df['f1'].iloc[select_row_test]
        prec_train_list = model_scores_df['precision'].iloc[select_row_train]
        prec_test_list = model_scores_df['precision'].iloc[select_row_test]
        acc_train_list = model_scores_df['accuracy'].iloc[select_row_train]
        acc_test_list = model_scores_df['accuracy'].iloc[select_row_test]

        ax2.plot(scanList, rec_train_list, 'ro--', label="Recall Train")
        ax2.plot(scanList, rec_test_list, 'ro-', label="Recall Test")
        ax2.plot(scanList, f1_train_list, 'go--', label="F1-score Train")
        ax2.plot(scanList, f1_test_list, 'go-', label="F1-score Test")
        ax2.plot(scanList, prec_train_list, 'yo--', label="Precision Train")
        ax2.plot(scanList, prec_test_list, 'yo-', label="Precision Test")
        ax2.plot(scanList, acc_train_list, 'bo--', label="Accuracy Train")
        ax2.plot(scanList, acc_test_list, 'bo-', label="Accuracy Test")
        
        ax2.set_xlabel(scanParam, fontsize=14)
        ax2.set_ylabel('Metric Value', fontsize=14)
        ax2.set_title(f"Evaluation Metrics for Data {dataNumber}, Scan '{scanParam}' ", fontsize=14)
        ax2.legend(loc='auto', fontsize=14)
        ax2.set_ylim([0, 1.05])
    
    
    if save and plot:
        plt.savefig(f'figures/LogLoss_Metrics_d{dataNumber}_{scanParam}.png')

    return model_scores_df
       

    
def compare_models(dataNumber, X_tr, y_tr, X_te, y_te, model_list, model_names_list, title, plot=1, save=0):

    """
    This is a function to compare models. 
    It creates  ROC curve and calculate metrics: 
    Returns to dataframe of listing evaluation metrics.
    
    dataNumber: # for Data file in use (1, 2, 3, 4, 5)
    X_tr: training data
    y_tr: training labels
    X_te: testing data
    y_te: testinglabels
    model_list: list of the models
    model_names_list: list of the title of the models
    title: title of comparison plot
    plot: Bool parameter to control creating the plot
    save: Bool parameter to control saving the plot
    """
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        cp = sns.color_palette()
    
    model_scores_list = []
    
    for i,clf in enumerate(model_list):
        
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

        
        fit_scores_train = {'Params': f'{model_names_list[i]}  Train ',
                        'precision': precision_train,
                        'recall': recall_train,
                        'f1': f1_train,
                        'accuracy': accuracy_train,
                        'auc': auc_train,
                           }
    
        fit_scores_test = {'Params': f'Test',
                        'precision': precision_test,
                        'recall': recall_test,
                        'f1': f1_test,
                        'accuracy': accuracy_test,
                        'auc': auc_test,
                          }
    
        model_scores_list.append(fit_scores_train)
        model_scores_list.append(fit_scores_test)
    
        if plot:
            #label_train = f"{model_names_list[i]} Train: prec={precision_train}, rec={recall_train}, f1={f1_train}, acc={accuracy_train}, AUC={auc_train}"
            #label_test = f"{model_names_list[i]} Test: prec={precision_test}, rec={recall_test}, f1={f1_test}, acc={accuracy_test}, AUC={auc_test}"
            label_train = f"{model_names_list[i]} Train"
            label_test = f"{model_names_list[i]} Test"
            ax.plot(fpr_train, tpr_train, lw=2, color=cp[i], linestyle='dashed', label=label_train)
            ax.plot(fpr_test, tpr_test, lw=2, color=cp[i], label=label_test)  


    model_scores_df = pd.DataFrame(model_scores_list)
    model_scores_df = model_scores_df.set_index('Params')
    #print(model_scores_df)
            
    if plot:    
        #ax.plot([0, 1], [0, 1], color='0.7', lw=2, linestyle='-.')
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f"ROC Curve for Data {dataNumber}, Model Comparison", fontsize=14)
        ax.legend(loc='auto', fontsize=13)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.5, 1.05])
          
        if save:
            plt.savefig(f'figures/ROC_modelCompare_{title}_d{dataNumber}.png')

    return model_scores_df

    

def plot_scan6_xgb(dataNumber, X_tr, y_tr, X_te, y_te, xgbParams, scanParam, scanList, weights, save=0):

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
    weights: Bool parameter to use sample_weights or not.
    save: Bool parameter to control saving the plot
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    #plt.tight_layout(pad=5)
    
    model_scores_list = []
    
    weigths_train = None
    
    if weights:
        print('Sample weights are used!\n--------\n')
        weigths_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_tr)
    
    for ax, s in zip(axes.flat, scanList):
        
        xgbParams[scanParam]=s
        
        clf = XGBClassifier(**xgbParams)
          
        clf.fit(X_tr, y_tr, sample_weight=weigths_train)
    
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
        ax.plot([0, 1], [0, 1], color='0.7', lw=2, linestyle='-.')
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
    
    