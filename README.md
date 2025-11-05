# FinalReport
Applications of DataScience-48479853-TabPFN 
This repository consists of scripts and the data required to replicate the core behaviour of the TabPFN on small tabular tasks and evaluate TabPFN on a new and self constructed student performance dataset.
Installation step:
Direct installatioin from the source:
# pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
and then install the requirements provided.



The data like breast cancer, Iris and wine datasets were already loaded in the provided scripts so there is no need to download.
Example final output of metrics comparision:
DecisionTree (baseline) : Test Set Metrics 
AUC:       0.707
Accuracy:  0.671
F1:        0.675
Precision: 0.667
Recall:    0.683
Confusion Matrix [ [TN FP] ; [FN TP] ]:
[[79 41]
 [38 82]]

DecisionTree (baseline) : Stratified 5-fold CV
AUC: mean=0.702  std=0.040
ACC: mean=0.665  std=0.016

TabPFN : Test Set Metrics 
AUC:       0.744
Accuracy:  0.683
F1:        0.678
Precision: 0.690
Recall:    0.667
Confusion Matrix [ [TN FP] ; [FN TP] ]:
[[84 36]
 [40 80]]

TabPFN : Stratified 5-fold CV 
AUC: mean=0.738  std=0.044
ACC: mean=0.674  std=0.026



