# Pre_Surv_COVID_19
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3766350.svg)](https://doi.org/10.5281/zenodo.3766350)\
The sudden increase of COVID-19 cases is putting a high pressure on health-care services worldwide. At the current stage, fast, accurate and early clinical assessment of the disease severity is vital. To support decision making and logistical planning in healthcare systems, this study leverages a database of blood samples from 485 infected patients in the region of Wuhan, China to identify crucial predictive biomarkers of disease mortality. For this purpose, machine learning tools selected three biomarkers that predict the mortality of individual patients with more than 90% accuracy: lactic dehydrogenase (LDH), lymphocyte and high-sensitivity C-reactive protein (hs-CRP). In particular, relatively high levels of LDH alone seem to play a crucial role in distinguishing the vast majority of cases that require immediate medical attention. This finding is consistent with current medical knowledge that high LDH levels are associated with tissue breakdown occurring in various diseases, including pulmonary disorders such as pneumonia. Overall, this paper suggests a simple and operable decision rule to quickly predict patients at the highest risk, allowing them to be prioritised and potentially reducing the mortality rate.

## Version

### Platform

Windows10

### Library 

- python==3.6
- pandas==1.0.1
- matplotlib==3.1.3
- numpy==1.18.1
- sklearn==0.21.2
- xgboost==0.90
- graphviz==0.13.2
- seaborn==0.10.0

## Folder structure
```text
.
│  EDA.py                                           # Exploratory Data Analysis
│  preprocess.py                                    # Data preprocessing
│  README.md                                        # This document
│  Main_of_features_selection                       # features_selection
|  utils_features_selection                         # Universal tool for features_selection
│  utils.py                                         # Universal tool
data
        time_series_375_prerpocess.xlsx             # Training data: 375 patients
        time_series_375.parquet                     # Training data after format conversion
        time_series_test_110_preprocess.xlsx        # Testing data: 110 patients
        time_series_test_110.parquet                # Testing data after format conversion
```

# Usage

## Main_of_features_selection.py

1.`selected_cols = features_selection()` can obtain Supplementary Figure 1, Supplementary Figure 2, Table 3:

2.`single_tree()` can obtain Supplementary Table 1, Supplementary Table 2, Figure 2, Table 4

3.`Compare_with_other_method(selected_cols)` can obtain Supplementary Figure 4, Supplementary Table 3



## EDA.py

1.`predicted_time_horizon()` can obtain Figure 3's B and C

2.`decision_tree_top3_feats_predict_result()` can obtain Table 4, Supplementary Figure 5, Supplementary Figure 6, Supplementary Figure 7

3.`plot_f1_time_single_tree_test_train()` can obtain Figure 3's D and E



## Citation
Yan, L., Zhang, H., Goncalves, J. et al. An interpretable mortality prediction model for COVID-19 patients. Nat Mach Intell 2, 283–288 (2020). https://doi.org/10.1038/s42256-020-0180-7


## Contact
Prof. Ye Yuan, yye@hust.edu.cn





