# D.O.M.E. (Detection Of Misstatements Engine)
 
D.O.M.E. is a data product which takes in financial statements and identifies whether the financial statements are misstated or not.

## Who can potentially benefit?
The following groups can benefit from our data product:
* Auditors can get a better idea of which corporations are more likely to missrepresent their fiscal outlook by filing misstatements.
* Investors can be aware of misstatement risks before investing in any corporation

## How it works?
D.O.M.E. uses machine learning and big data analytics to classify any financial statement as either misstated or not misstated with about 82% accuracy.


## Results

Go to https://github.com/chiu/accounting-ml-project/blob/master/machine_learning/rf_and_logistic_notebook.ipynb
    
`rf_and_logistic_notebook.ipynb` new version of random forest and logistic regression with results printed. 

## How to Run:
* Copy the file `rf_and_logistic.py` from:
* https://github.com/chiu/accounting-ml-project/blob/master/machine_learning/rf_and_logistic.py
* Copy code onto the SFU cluster.
* Run the following command: `spark-submit rf_and_logistic.py`


## Table of Contents:

```
.
├── LICENSE
├── README.md
├── data
├── data_integration
│   ├── DGLS_sheets_integration.ipynb
│   ├── aaer_labeling.py
│   ├── data_integration.py
│   └── ibes_integration_fix-Copy1.ipynb
├── eda
│   ├── AAPL.png
│   ├── MSFT.png
│   ├── Visualise_PCA_clusters.ipynb
│   ├── comparing_std_summ_std.ipynb
│   ├── corr_matrix_plot_2-Copy1\ (1).ipynb
│   ├── heatmap_correlation_matrix.png
│   ├── industry_wise_segmentation.py
│   ├── num_aaer_per_firm-Copy1.ipynb
│   ├── reasons_for_misstatement-Copy2.ipynb
│   ├── timeseries.html
│   └── timeseries.py
├── experimental
│   └── one_hot_encoding_experiment.py
├── machine_learning
│   ├── clustering_kmeans.py
│   ├── old_version_rf_and_logistic.ipynb
│   ├── performance_metricslogistic_with_validation.csv
│   ├── performance_metricslogisticregression.csv
│   ├── performance_metricslogisticregressionwithbestthreshold.csv
│   ├── performance_metricsrandomforest.csv
│   ├── performance_metricsrf_with_validation.csv
│   ├── rf_and_logistic.py
│   └── rf_and_logistic_notebook.ipynb
├── obsolete
│   ├── data_integration_obsolete
│   │   ├── attempt_1hot.py
│   │   ├── load_compustat.py
│   │   └── load_integrated_data.py
│   ├── machine_learning_obsolete
│   │   ├── logistic_balancing_weights.ipynb
│   │   ├── misstatement_detection-Copy15.ipynb
│   │   ├── misstatement_detection-Copy15.py
│   │   ├── nn-example.ipynb
│   │   ├── nn_integrated.py
│   │   └── nn_trial_2.py
│   ├── nullcount
│   │   ├── _SUCCESS
│   │   └── part-00000-e4758198-5946-4939-8f03-746415fb32de-c000.csv
│   ├── preprocessing
│   │   └── pca_take2-Copy2.ipynb
│   └── tableau_charts
│       └── Misstated\ count\ analysis(industrywise).twbx
├── poster
│   ├── Screenshot-2018-3-30\ rf_and_logistic_v2-Copy1.png
│   ├── cmpt733_vcs_poster_v1.pdf
│   ├── data_pipeline_flow.png
│   ├── data_pipeline_flow_v2.png
│   ├── heatmap_correlation_matrix.png
│   ├── methodology_final.jpg
│   ├── misstatements_per_industry.png
│   ├── num_aaer_vs_reason.png
│   ├── num_corp_with_misstatements.png
│   ├── pcaPlot.png
│   ├── pcaPlot2.png
│   ├── pca_plot.png
│   ├── poster-733\ (1)
│   │   ├── SFUBigData_logo.jpg
│   │   ├── beamerposter.sty
│   │   ├── beamerthemeconfposter.sty
│   │   ├── heatmap_correlation_matrix.png
│   │   ├── logo.png
│   │   ├── main.tex
│   │   ├── matplotlib.svg
│   │   ├── misstatements_per_industry.png
│   │   ├── num_aaer_vs_reason.png
│   │   ├── num_corp_with_misstatements.png
│   │   ├── pcaPlot.png
│   │   ├── pcaPlot2.png
│   │   ├── placeholder.jpg
│   │   ├── sample.bib
│   │   ├── tableau_viz.pdf
│   │   ├── v2_word_cloud_logistic_regression.png
│   │   ├── v3_word_cloud_logistic_regression.png
│   │   └── word_cloud_logistic_regression.png
│   ├── poster-733\ (10).pdf
│   ├── poster.pdf
│   ├── poster_for_cornerstone_printing_vc.pdf
│   ├── poster_for_printing_staples.pdf
│   ├── poster_presentation_final.pdf
│   ├── screenshot-2018-3-30_rf_and_logistic_v2-copy1_480.png
│   ├── tableau_viz.pdf
│   ├── v2_word_cloud_logistic_regression.png
│   ├── v3_word_cloud_logistic_regression.png
│   ├── word\ cloud.png
│   └── word_cloud_logistic_regression.png
├── report
│   └── report.pdf
└── slides
    ├── Detecting\ Misstatements\ v2.pptx
    ├── Detecting\ Misstatements.pptx
    └── random_forest_medium.png
```

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

`corr_matrix_plot_2-Copy1 (1).ipynb`: code for making correlation matrix heat map

`num_aaer_per_firm-Copy1.ipynb`: finding number of firms with aaer

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


