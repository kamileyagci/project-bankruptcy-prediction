# Bankruptcy Prediction for Polish Companies

**Author: Kamile Yagci**

**Blog URL: https://kamileyagci.github.io/**


## Overview

In this study, I will analyze the financial standings of the Polish companies to identify whether the company will banckrupt in 1-5 years. I will use the Ensemble Method 'XGBoost', eXtreme Gradient Boosting, for classification. 


<a href="//commons.wikimedia.org/wiki/File:Panorama_siekierkowski.jpg" title="Panorama Warszawy z mostu Siekierkowskiego, 2020"><img src="/figures/Panorama_siekierkowski.jpeg"/></a>


## Repository Content

* data: directory containing the data files
* figures: directory containing figures/images
* notUsed: directory containing some prelimenary analysis, which is not part of the latest version of the project
* saved_models: directory containing the saved models
* .gitignore: text file that contains the list of files/directories that should not be tracked by git repository
* README.md: markdown file that describes the git repository and the project
* analysis_1_explore.ipynb: jupyter notebook for analysis part 1, data exploring
* analysis_2_imbalance.ipynb: jupyter notebook for analysis part 2, class imbalance study for Data 3
* analysis_3_data3.ipynb: jupyter notebook for analysis part 3, XGBoost Classification for Data 3
* analysis_4_dataAll.ipynb: jupyter notebook for analysis part 4, XGBoost Classification for all data sets AND Interpretation of Results
* functions.py: python file containing the functions used in the analysis


## Project Outline

* Business Problem
* Data
* Methods
* Analysis and Results
    * Class Imbalance Study for Dataset 3
        * ???
    * XGBoost Classification on Dataset 3
        * ???
    * XGBoost Classification on all five datasets
        * ??
        * Final Model
* Conclusion
* Future Work


## Business Problem

KPMG, international corporate financial consulting firm, hired me to analyze the financial standing of the Polish companies. The goal of the anlaysis is identifying whether the business will go to bankruptcy in 1-5 years or not. KMPG will use the results of this study to provide an early warning to Polish business clients on their financial standings, so they can take preventive actions.


## Data

The data contains the financial information and bankruptcy status of Polish companies. It is collected from Emerging Markets Information Service (EMIS, [Web Link]).

The data was collected in the period of
* 2000-2012 for the bankrupt companies
* 2007-2013 for the still operating companies

Depending on the forecasting period, the data is classified in five categories/datasets.:

* 1st Year: the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years (1year.arff).
* 2nd Year: the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years (2year.arff).
* 3rd Year: the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years (3year.arff).
* 4th Year: the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years (4year.arff). 
* 5th Year: the data contains financial rates from 5th year of the forecasting period and corresponding class label that indicates bankruptcy status after 1 year (1year.arff).

UCI Link: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

In my analysis, I name the five data files as:

* Data 1: '1year.arff'
* Data 2: '2year.arff'
* Data 3: '3year.arff'
* Data 4: '4year.arff'
* Data 5: '5year.arff'

The number of companies in each dataset and class distributions:

| Data # | Total | Still Operating (class=0) | Bankrupt (class=1) |
| :- | -: | :-: | :-: |
| Data 1 | 7027 | 6756 | 271 
| Data 2 | 10173 | 9773 | 400 
| Data 3 | 10503 | 10008 | 495
| Data 4 | 9792 | 9277 | 515
| Data 5 | 5910 | 5500 | 410



## Method

I will use Ensemble Method 'XGBoost', eXtreme Gradient Boosting, for classification. 

This is a binary classification problem, since my goal is to identify whether the company will bankrupt or not. 

I will focus on the performance of 'recall' metric in order to minimize false negatives. Besides, I will also keep an eye on 'f1', and 'AUC' metrics.


## Analysis and Results

Initially, I explored the XGBoost Classifier models on Data 3. After determing the best model designs, I applied them on other datasets and compare the results.

Note: No cleaning applied to data. XGBoost Classifier can handle the missing values and outliers.

### Class Imbalance

The class imbalance is one of the main issues in this data.

Imbalance Ratio = (# of class 0 companies) / (# of class 1 companies)

| Data # | Imbalance Ratio | Sqrt of Imbalance Ratio |
| :- | -: | :-: |
| Data 1 | 24.93 | 4.99 
| Data 2 | 24.43 | 4.94 
| Data 3 | 20.22 | 4.50
| Data 4 | 18.01 | 4.24 
| Data 5 | 13.41 | 3.66 

There are two approaches to deal with the class imbalance. I have used both approaches together, since it provided a better result.

* sample_weight: parameter when training the data.
    * The weights for training sample are calculated for each dataset seperately and used when during training.

* scale_pos_weight: parameter when initiating the classifier 
    * I provide certain values to initiate the classifier. I either use the imbalance ratio or square root of the imbalance ratio. These values are not exactly same for the datasets, but close enough to use a constant about average number.
    * Optimized values:
        * max_depth=4: scale_pos_weight=4.5 (~square root of imbalance ratio)
        * max_depth=5: scale_pos_weight=20 (~imbalance ratio)
        * max_depth=6: scale_pos_weight=20 (~imbalance ratio)


## Conclusion

5-year Period (Data 1):
    * Model successfully identifies the 80.4 of the true bankrupt companies, which will bankrupt 5 years later. (recall)
    * Among the model predicted bankruptcy companies, 64.1% of them are true bankrupt companies, which will bankrupt 5 years later. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 71.3%.
    
4-year Period (Data 2):
    * Model successfully identifies the 62.0 of the true bankrupt companies, which will bankrupt 4 years later. (recall)
    * Among the model predicted bankruptcy companies, 50.6% of them are true bankrupt companies, which will bankrupt 4 years later. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 55.7%.
    
3-year Period (Data 3):
    * Model successfully identifies the 72.0 of the true bankrupt companies, which will bankrupt 3 years later. (recall)
    * Among the model predicted bankruptcy companies, 53.5% of them are true bankrupt companies, which will bankrupt 3 years later. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 61.4%.

2-year Period (Data 4):
    * Model successfully identifies the 68.0 of the true bankrupt companies, which will bankrupt 2 years later. (recall)
    * Among the model predicted bankruptcy companies, 55.6% of them are true bankrupt companies, which will bankrupt 2 years later. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 61.1%.
    
1-year Period (Data 5):
    * Model successfully identifies the 78.9 of the true bankrupt companies, which will bankrupt 1 years later. (recall)
    * Among the model predicted bankruptcy companies, 60.7% of them are true bankrupt companies, which will bankrupt 1 years later. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 68.6%.

On Average:
    * Model successfully identifies the 72.3 of the true bankrupt companies. (recall)
    * Among the model predicted bankruptcy companies, 56.9% of them are true bankrupt companies. (precision)
    * The Harmonic Mean of Precision and Recall (f1-score) is 63.6%.
    
**Best common predictors**

* X27 profit on operating activities / financial expenses
* X34 operating expenses / total liabilities
* X5 [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365
 


## Future Work

* The model performance is not very good and overfitting is large. Search for alternative methods to improve performance.

* Each dataset can be optimized (with parameter tuning) seperately and create 5 different models, instead of one model. This will increase the overall performance.

* Created functions are long and repeating. Update them.

