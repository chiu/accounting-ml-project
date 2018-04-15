# D.O.M.E. (Detection Of Misstatements Engine)
 
D.O.M.E. is a data product which takes in financial statements and identifies whether the financial statements are misstated or not.

## Who can potentially benefit?
The following groups can benefit from our data product:
* Auditors can get a better idea of which corporations are more likely to missrepresent their fiscal outlook by filing misstatements.
* Investors can be aware of misstatement risks before investing in any corporation

## How it works?
D.O.M.E. uses machine learning and big data analytics to classify any financial statement as either misstated or not misstated with about 82% accuracy.

## How to Run:
* Copy the file `rf_and_logistic.py` from:
* https://github.com/chiu/accounting-ml-project/blob/master/machine_learning/rf_and_logistic.py
* Copy code onto the SFU cluster.
* Run the following command: 
  `spark-submit rf_and_logistic.py`


## Table of Contents:
* data: data from here has been emptied out, data can be found on SFU cluster under hdfs in `/user/vcs/`
* data_integration: code for integrating financial reports from CompuStat with AAER and IBES data. 
* eda: code for heatmap, number of misstatements per industry plots. 
* experimental: sandbox for code
### machine_learning: location of the logistic regression and random forest code. Contains the following:
#### rf_and_logistic.py
#### clustering_kmeans.py

* nullcount: file containing the number of null observations for each feature attribute
* poster: materials for poster
* preprocessing: code for preprocessing
* report: materials for report
* slides: materials for slides which were made for the video. 



