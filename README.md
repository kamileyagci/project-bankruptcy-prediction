# Bankruptcy Prediction for Polish Companies

**Author: Kamile Yagci**

**Blog URL: https://kamileyagci.github.io/**


## Overview

In this study, I will analyze the financial standings of the Polish companies to identify whether the company will banckrupt in 1-5 years. I will use the Ensemble Method 'XGBoost', eXtreme Gradient Boosting for classification. 


<a href="//commons.wikimedia.org/wiki/File:Panorama_siekierkowski.jpg" title="Panorama Warszawy z mostu Siekierkowskiego, 2020"><img src="/figures/Panorama_siekierkowski.jpeg"/></a>


## Repository Content

* data: directory containing the data files
* figures: directory containing figures/images
* notUsed: directory containing some prelimenary analysis, which is not part of the latest version of the project
* saved_models: directory containing the saved models
* .gitignore: text file that contains the list of files/directories that should not be tracked by git repository
* README.md: markdown file that described the git repository and project
* analysis_1_explore.ipynb: jupyter notebook for analysis part 1, data exploring
* analysis_2_imbalance.ipynb: jupyter notebook for analysis part 2, class imbalance study
* analysis_3_data3.ipynb: jupyter notebook for analysis part 3, XGBoost Classification for Dataset 3 '3year.arff' 
* analysis_4_dataAll.ipynb: jupyter notebook for analysis part 4, XGBoost Classification for all data sets AND Interpretation of Results
* functions.py: python file containing the functions used in the analysis



## Project Outline

* Business Problem
* Data
* Methods
* Analysis and Results
    * Baseline Model
    * Baseline + Regularization Model
    * Baseline + Dropout Layers Model
    * Baseline Model + Augmentation
    * New Model with more Layers and optimizer='Adam'
    * Train on whole dataset, Baseline Model with optimizer='Adam'
    * Final Model
* Conclusion
* Next Steps


## Business Problem

KPMG, international corporate financial consulting firm, hired me to analyze the financial standing of the Polish companies. The goal of the anlaysis is identifying whether the business will go to bankruptcy in 1-5 years or not. KMPG will use the results of this study to provide an early warning to Polish business clients on their financial standings, so they can take preventive actions.


## Data


## Method

I will focus on the performance of 'recall' metric in order to minimize false negatives. Besides, I will also keep an eye on 'f1', and 'AUC' metrics.


## Analysis and Results


## Conclusion


## Future Work



