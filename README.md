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
* Run the following command: `spark-submit rf_and_logistic.py`


## Table of Contents:

### data: 

    data from here has been emptied out, data can be found on SFU cluster under hdfs in `/user/vcs/`


### data_integration: 

    code for integrating financial reports from CompuStat with AAER and IBES data.

    `data_integration.py`: merging of annual, aaer and ibes dataset. 

    `aaer_labelling.py` contains custom udf function for labelling records as as misstatement or not misstatement. used for creating our class label. 
    
    `ibes_integration_fix-Copy1.ipynb` fix for bug involving joining annual with ibes which incorrect join conditions. 
    
 
 
### eda: 

    code for heatmap, number of misstatements per industry plots. 

    `AAPL.png`			
    
    `comparing_std_summ_std.ipynb`
        
    `industry_wise_segmentation.py`: num corporations with misstatement chart	
    
    `timeseries.html`
    
    `MSFT.png`		
    
    `corr_matrix_plot_2-Copy1` (1).`ipynb`: code for making correlation matrix heat map
    
    `num_aaer_per_firm-Copy1`.`ipynb`: finding number of firms with aaer
    
    `timeseries.py`: code for time series plots involving Earnings per Share
    
    `Visualise_PCA_clusters.ipynb`: visualization for PCA clusters
    
    `heatmap_correlation_matrix.png`		
    
    `reasons_for_misstatement-Copy2`.`ipynb`



### machine_learning: location of the logistic regression and random forest code. 
   
    `rf_and_logistic_notebook.ipynb` new version of random forest and logistic regression with results printed. 
    
    `rf_and_logistic.py` code for random forest and logistic regression
    
    `clustering_kmeans.py` code for doing kmeans clustering on pca components
    
    `old_version_rf_and_logistic.ipynb` old version of random forest and logistic regression with results printed. 


### experimental: sandbox for code

### obsolete: 

    `nullcount` folder containing the number of null observations for each feature attribute
    
    
### poster: materials for poster
### report: materials for report
### slides: materials for slides which were made for the video. 




