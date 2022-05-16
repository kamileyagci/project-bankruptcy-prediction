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

The data contains the financial information and bankruptcy status of Polish companies. 

The data is collected from Emerging Markets Information Service (EMIS, [Web Link]).

Data was collected in the period of

    * 2000-2012 for the bankrupt companies
    * 2007-2013 for the still operating companies

Depending on the forecasting period, dataset is classified in five categories/files:

* 1stYear: the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years.
* 2ndYear: the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years.
* 3rdYear: the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years.
* 4thYear: the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years. 
* 5thYear: the data contains financial rates from 5th year of the forecasting period and corresponding class label that indicates bankruptcy status after 

In my analysis, I name the five data files as:

* Data 1: 1year.arff 
* Data 2: 2year.arff 
* Data 3: 3year.arff
* Data 4: 4year.arff
* Data 5: 5year.arff

UCI Link: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data


## Method

I will focus on the performance of 'recall' metric in order to minimize false negatives. Besides, I will also keep an eye on 'f1', and 'AUC' metrics.


## Analysis and Results


## Conclusion


## Future Work



