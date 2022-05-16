# Bankruptcy Prediction for Polish Companies

**Author: Kamile Yagci**

**Blog URL: https://kamileyagci.github.io/**


<a href="<//commons.wikimedia.org/wiki/File:Panorama_siekierkowski.jpg>" title="Panorama Warszawy z mostu Siekierkowskiego, 2020"><img src="/figures/Panorama_siekierkowski.jpeg"/></a>

https://commons.wikimedia.org/wiki/User:Jmabel

<a href="//commons.wikimedia.org/wiki/File:Panorama_siekierkowski.jpg" title="Panorama Warszawy z mostu Siekierkowskiego, 2020"><img src="/figures/Panorama_siekierkowski.jpeg"/></a>


## Overview

In this study, I will analyze the financial standings of the Polish companies to identify whether the company will banckrupt in 1-5 years. I will use the Ensemble Method 'XGBoost', eXtreme Gradient Boosting for classification. 


## Business Problem

KPMG, international corporate financial consulting firm, hired me to analyze the financial standing of the Polish companies. The goal of the anlaysis is identifying whether the business will go to bankruptcy in 1-5 years or not. KMPG will use the results of this study to provide an early warning to Polish business clients on their financial standings, so they can take preventive actions.


Qbolewicz, CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0>, via Wikimedia Commons

I will focus on the performance of 'recall' metric in order to minimize false negatives. Besides, I will also keep an eye on 'f1', and 'AUC' metrics.