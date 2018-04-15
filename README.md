# D.O.M.E. (Detection Of Misstatement Engine)
 
D.O.M.E. is a data product which takes in financial statements and identifies whether the financial statements are misstated or not.

## Who can be potentially benefit?
The following groups can be benefit from our data product:
* Auditors can get a better idea of which corporations are more likely to missrepresent the fiscal outlook by filing misstatements.
* Investors can be aware of misstatement risks before investing in any corporation

## How it works?
D.O.M.E. uses machine learning and big data analytics to classify whether any financial statement is misstated or not with about 82% accuracy.

## How to Run:
* Copy the file rf_and_logistic.py from:
* https://github.com/chiu/accounting-ml-project/blob/master/machine_learning/rf_and_logistic.py

* Copy code onto the SFU cluster.

* Run the following command: 

* spark-submit rf_and_logistic.py

## Table of Contents:
* data
* data_integration: code for integrating financial reports with AAER and IBES data. 
* eda: code for heatmap, number of misstatements per industry plots. 
* experimental: sandbox for code
* machine_learning: location of the logistic regression and random forest code. 
* nullcount
* poster
* preprocessing: code for preprocessing
* report
* slides



