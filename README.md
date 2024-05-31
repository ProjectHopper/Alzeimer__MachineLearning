# Project Overview

Predictive modeling for Alzheimer classification: </BR>
Comparative analysis of machine learning algorithms and ensemble techniques.</BR>
Alzheimer's disease is a neurodegenerative disorder that affects memory and </BR>
behavior. Since there is currently no cure available, early-stage prevention is crucial. </BR>
In this project, I will use heatmaps, logistic regression, decision trees, k-nearest </BR>
neighbors (k-NN), and random forests to predict Alzheimer's disease and identify </BR>
the features that most significantly aƯect its development. I will also compare the </BR>
accuracy and confusion matrices of these models to determine the best </BR>
classification method for this task. </BR></BR>

## Features: 
o Age </BR>
o Years of education (EDUC) </BR>
o Socioeconomic status (SES) </BR>
o Clinical Dementia Rating (CDR) </BR>
     0: none </BR>
     0.5: possible </BR>
     1: positive </BR>
o Mini-Mental State Examination (MMSE) </BR>
o Labels:</BR>
    Group: nondemented (0: negative), converted (1: positive), demented (1: positive)</BR>
    - Methodology </BR>
1. Load data: </BR>
Load data from the CSV file. (alzheimer.csv) </BR>
2. Data preprocessing: </BR>
  Combine group labels into binary categories. </BR>
  Nondemented-0, Converted -0.5, Demented 1 </BR>
  Nondemented-0, Converted & Demented – 1 </BR>
  Use random 50/50 splits for training and testing data for generalizability, avoiding overfitting. </BR>
3. Visualization </BR>
Supervised learning classification using logistic regression, Decision Trees, </BR>
k-NN and Random Forest for classification and regression modeling. </BR>
  Heatmap</BR>
  Logistic Regression </BR>
  Decision Trees</BR></BR>

## Results </BR>
 Heatmap: correlation matrix of features </BR>
 0-Nondemented Correlation Matrix: 0.02 Age & EDUC </BR>
0.5-Converted Correlation Matrix: 0.00 MMSE & EDUC </BR>
1-Demented Correlation Matrix: 0.03 Age & CDR, SES & MMSE</BR>
Overall, Age, EDUC, and MMSE mentioned 2 times. </BR>
Age, years of education, and mental state are the main factors contributing to Alzheimer's disease. </BR>
Error rate (Random Forest): The best n is 10 and d is 5, with an error rate of 6.78%. </BR></BR>
# Conclusion </BR>
Random Forest is the most eƯective model for predicting Alzheimer’s disease in the dataset. Heatmap analysis revealed correlations between 
features, which were analyzed based on their labels. The heatmap showed that age, years of education, and mental state were consistently mentioned 
across the correlation matrices. Random Forest determined that the bestcombination of hyperparameters was n=10, d=5, yielding an error rate of 6.78%.
Early-stage detection is critical in Alzheimer's disease, and based on this project, I discovered that the converted stage correlates with mental status
and years of education. I believe that prolonged exposure to intense educational environments may lead to higher levels of mental stress,
potentially contributing to the disease. Demented status, age, clinical dementia rating, and socioeconomic status were also found to be significant 
factors. It is noteworthy that Alzheimer's patients are mostly found in older age groups. Moreover, the combination of socioeconomic status and mental
state is intriguing, suggesting that lower socioeconomic status may exacerbate mental stress levels, potentially contributing to dementia. 
</BR>
</BR>
**Pre-requisite libraries**</BR>

* `pip install pandas` </BR>
* `pip install scikit-learn` </BR>
* `pip install matplotlib`</BR>
* `pip install seaborn`</BR>
* `pip install numpy` </BR></BR>

## How to install the programs

Open a terminal or power shell</BR>
Navigate to the project saved location</BR>
Run the commands</BR></BR>

## Expected output
Lwon_final_project.py file is the mail file that will present charts and graphs. 



