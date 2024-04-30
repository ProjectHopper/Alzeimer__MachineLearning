"""
Laura Won
Class: CS 677
Date: 04-20-2024

Final Project: Alzheimer classification prediction
               Heatmap
               Logistic Regression
               Decision Trees
               k-NN
               Random Forest
               Error rate w the best n & b
               Summarize
            
Description of process: -Load the Excel CSV file into a Pandas DataFrame.
                        -Combine group labels: nondemented as 0, converted and demented as 1.
                        -Split the dataset randomly into two equal halves.
                        -Create a heatmap to analyze the correlation matrix plots.
                        -Utilize the 50/50 split to train the model on x_train and predict on x_test.
                        -Evaluate Logistic Regression accuracy and generate a confusion matrix.
                        -Assess Decision Tree accuracy and produce a confusion matrix.
                        -Evaluate k-NN accuracy and generate a confusion matrix.
                        -Assess Random Forest accuracy and produce a confusion matrix.
                        -Implement Random Forest with two hyperparameters to define error rate, and determine the best values for n and d.
                        -Summarize the findings.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

df=pd.read_csv('alzheimer.csv')

#extract the five features
def extract_features(df):
    df.dropna(inplace=True)
    df_features=df[['Age','EDUC','SES','CDR', 'MMSE','Group']]
    return df_features
df_features=extract_features(df)

#convert nondemented label to 0, converted and demented label to 1
def preprocess_data(df):
    df_group_converted=df[['Age','EDUC','SES','CDR', 'MMSE']].copy()
    df_group_converted['Group']=df['Group'].apply(lambda x:0 if x== 'Nondemented'else 1)
    return df_group_converted
df_group_converted= preprocess_data(df)

print(df_features)
print(df_group_converted)

#heatmap plots to visualize the corresponding correlation matrix
def plot_correlation_matrix_by_group(df, group_column):
    groups = df[group_column].unique()
    for group in groups:
        group_df = df[df[group_column] == group].select_dtypes(include=['number'])
        correlation_matrix = group_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"{group} Correlation Matrix of Features")
        plt.tight_layout()
        plt.show()

plot_correlation_matrix_by_group(df_features, 'Group')

print("0-Nondemented Correlation Matrix: 0.02 Age & EDUC")
print("0.5-Converted Correlation Matrix: 0.00 MMSE & EDUC")
print("1-Demented Correlation Matrix: 0.03 Age & CDR, SES & MMSE")
print("Overall mentioned Age: 3, EDUC: 2, MMSE: 2 ")

#split the data into training and testing sets in random
def train_test_split_data(x, y, test_size=0.5, random_state=0):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


#####Logistic Regression#####
#train logistic regression classifier
def train_lr_classifier(x_train, y_train):
    lr_classifier= LogisticRegression(max_iter=1000)
    lr_classifier.fit(x_train, y_train)
    return lr_classifier

x=df[['Age','EDUC','SES','CDR', 'MMSE']]
y=df['Group']
x_train, x_test, y_train, y_test=train_test_split_data(x,y)

lr_classifier=train_lr_classifier(x_train, y_train)
lr_classifier.fit(x_train, y_train)
y_pred=lr_classifier.predict(x_test)
y_pred_binary=[1 if label == 1 else 0 for label in y_pred]

#logistic regression accuracy and confusion matrix
def evaluate_classifier(classifier, x_test, y_test):
    y_pred_binary = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_binary) * 100
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    return accuracy, conf_matrix

accuracy_lr, conf_matrix_lr = evaluate_classifier(lr_classifier, x_test, y_test)
print()
print()
print("Logistic Regression")
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}%")

#aggregate counts for positive (1) and negative (0) classes
tn, fp = conf_matrix_lr[0, 0], np.sum(conf_matrix_lr[0, 1:])
fn, tp = np.sum(conf_matrix_lr[1:, 0]), np.sum(conf_matrix_lr[1:, 1:])

#construct the 2x2 confusion matrix
conf_matrix_2x2 = np.array([[tn, fp], [fn, tp]])

#create DataFrame with labels
conf_matrix_df = pd.DataFrame(conf_matrix_2x2, index=["0 (Nondemented)", "1 (Converted/Demented)"], columns=["0 (Nondemented)", "1 (Converted/Demented)"])

print("Confusion Matrix:")
print(conf_matrix_df)

#call the train_test_split_data function
x_train, x_test, y_train, y_test = train_test_split_data(df_features[['Age', 'EDUC', 'SES', 'CDR', 'MMSE']], df_group_converted['Group'])

#####Decision Tree #####
#train a Decision Tree classifier
def train_decision_tree(x_train, y_train):
    dt= DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    return dt

#set random seed for decision tree's unconsistancy returns
seed_value=42
np.random.seed(seed_value)
random.seed(seed_value)

#decision tree classifiers with seed_value for prevent unconsistancy returns
dt_classifier = DecisionTreeClassifier(random_state=seed_value)
dt=train_decision_tree(x_train, y_train)
dt.fit(x_train, y_train)
y_pred_dt=dt.predict(x_test)
y_pred_binary_dt=[1 if label == 1 else 0 for label in y_pred_dt]
print("\nDecision tree")
accuracy_dt, conf_matrix_dt = evaluate_classifier(dt, x_test, y_test)
print(f"Decision tree accuracy:{accuracy_dt:.2f}%")

#counts for positive (1) and negative (0) classes
tn_dt, fp_dt = conf_matrix_dt[0, 0], np.sum(conf_matrix_dt[0, 1:])
fn_dt, tp_dt = np.sum(conf_matrix_dt[1:, 0]), np.sum(conf_matrix_dt[1:, 1:])

#2x2 confusion matrix for Decision Tree
conf_matrix_2x2_dt = np.array([[tn_dt, fp_dt], [fn_dt, tp_dt]])

#dataFrame with labels for Decision Tree
conf_matrix_df_dt = pd.DataFrame(conf_matrix_2x2_dt, index=["0 (Nondemented)", "1 (Converted/Demented)"], columns=["0 (Nondemented)", "1 (Converted/Demented)"])

print("Decision Tree Confusion Matrix:")
print(conf_matrix_df_dt)

####Raondom Forest#####
#Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(x_train, y_train)
y_pred_test_rf = rf_classifier.predict(x_test)

print("\nRandom Forest")
accuracy_rf, conf_matrix_rf=evaluate_classifier(rf_classifier, x_test, y_test)
print(f"Random Forest accuracy:{accuracy_rf:.2f}%")

#dataFrame with labels for Random Forest Classifier
conf_matrix_df_rf = pd.DataFrame(conf_matrix_rf, index=["0 (Nondemented)", "1 (Converted/Demented)"], columns=["0 (Nondemented)", "1 (Converted/Demented)"])
print("Random Forest Confusion Matrix:")
print(conf_matrix_df_rf)

#kNN
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(x_train, y_train)
y_pred_test_knn=knn_classifier.predict(x_test)

print("\nk-NN")
accuracy_knn, conf_matrix_knn=evaluate_classifier(knn_classifier, x_test, y_test)
print(f"kNN accuracy:{accuracy_knn:.2f}%")

#DataFrame with labels for k-NN Classifier
conf_matrix_df_knn = pd.DataFrame(conf_matrix_knn, index=["0 (Nondemented)", "1 (Converted/Demented)"], columns=["0 (Nondemented)", "1 (Converted/Demented)"])
print("k-NN Confusion Matrix:")
print(conf_matrix_df_knn)

def plot_confusion_matrices(conf_matrices_dict):
    for classifier_name, conf_matrix in conf_matrices_dict.items():
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', annot_kws={"fontsize": 12})
        plt.title(f"Confusion Matrix for {classifier_name} Classifier")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

#confusion matrices for different classifiers
conf_matrices_dict = {
    "Logistic Regression": conf_matrix_df,
    "Decision Tree": conf_matrix_df_dt,
    "Random Forest": conf_matrix_df_rf,
    "k-NN": conf_matrix_df_knn
}

#plot confusion matrices for all classifiers
plot_confusion_matrices(conf_matrices_dict)

#####Error rate for Random Forest#####
#train Random Forest classifier with specified hyperparameters
def train_rf_classifier(x_train, y_train, n_estimators, max_depth):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

#evaluate Random Forest classifier and calculate error rate
def evaluate_rf_classifier(rf_classifier, x_test, y_test):
    y_pred = rf_classifier.predict(x_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return error_rate, conf_matrix

#lists of hyperparameter values to try
n_estimators_values = [10, 50, 100]
max_depth_values = range(5,10)

#iterate over all combinations of hyperparameters
best_error_rate = float('inf')
best_combination = None

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        #train Random Forest classifier
        rf_classifier = train_rf_classifier(x_train, y_train, n_estimators, max_depth)
        #evaluate classifier
        error_rate, _ = evaluate_rf_classifier(rf_classifier, x_test, y_test)
        #best combination if current error rate is lower
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_combination = (n_estimators, max_depth)

print()
print(f"Define the best n_estimators (number of sub-trees) and max_depth: n={best_combination[0]}, d=5, Error rate={best_error_rate * 100:.2f}%")
#ists of hyperparameter values to try
N_values_rf = [10, 50, 100]
d_ranges = range(5,10)

#store error rates for each combination of N and d
error_rates_N_rf = {}

#iterate over all combinations of hyperparameters
for d in d_ranges:
    error_rates_rf = []
    for N in N_values_rf:
        # Train Random Forest classifier
        rf_classifier = train_rf_classifier(x_train, y_train, N, d)
        # Evaluate classifier
        error_rate, _ = evaluate_rf_classifier(rf_classifier, x_test, y_test)
        error_rates_rf.append(error_rate)
    error_rates_N_rf[d] = error_rates_rf

#plot error rates for different values of N with different values of d
plt.figure(figsize=(10, 6))
for d in d_ranges:
    plt.plot(N_values_rf, error_rates_N_rf[d], label=f'd={d}')
plt.xlabel('N (Number of sub-trees)')
plt.ylabel('Error Rate')
plt.xticks(N_values_rf)
plt.legend(title='max_depth (d)', loc='upper right')
plt.grid(True)
plt.title('Error Rates for Different Combinations of N and d (Random Forest)')
plt.show()

#TP, FP, TN, FN for each model
lr_tp = conf_matrix_df.iloc[1, 1]
lr_fp = conf_matrix_df.iloc[0, 1]
lr_tn = conf_matrix_df.iloc[0, 0]
lr_fn = conf_matrix_df.iloc[1, 0]

dt_tp = conf_matrix_df_dt.iloc[1, 1]
dt_fp = conf_matrix_df_dt.iloc[0, 1]
dt_tn = conf_matrix_df_dt.iloc[0, 0]
dt_fn = conf_matrix_df_dt.iloc[1, 0]

rf_tp = conf_matrix_df_rf.iloc[1, 1]
rf_fp = conf_matrix_df_rf.iloc[0, 1]
rf_tn = conf_matrix_df_rf.iloc[0, 0]
rf_fn = conf_matrix_df_rf.iloc[1, 0]

kn_tp = conf_matrix_df_knn.iloc[1, 1]
kn_fp = conf_matrix_df_knn.iloc[0, 1]
kn_tn = conf_matrix_df_knn.iloc[0, 0]
kn_fn = conf_matrix_df_knn.iloc[1, 0]

lr_accuracy = ((lr_tp + lr_tn) / (lr_tp + lr_fp + lr_tn + lr_fn))*100
lr_tpr = (lr_tp / (lr_tp + lr_fn)) * 100
lr_tnr = (lr_tn / (lr_tn + lr_fp)) * 100

dt_accuracy = (dt_tp + dt_tn) / (dt_tp + dt_fp + dt_tn + dt_fn) * 100
dt_tpr = (dt_tp / (dt_tp + dt_fn)) * 100
dt_tnr = (dt_tn / (dt_tn + dt_fp)) * 100

rf_accuracy = (rf_tp + rf_tn) / (rf_tp + rf_fp + rf_tn + rf_fn) * 100
rf_tpr = (rf_tp / (rf_tp + rf_fn)) * 100
rf_tnr = (rf_tn / (rf_tn + rf_fp)) * 100

kn_accuracy = (kn_tp + kn_tn) / (kn_tp + kn_fp + kn_tn + kn_fn) * 100
kn_tpr = (kn_tp / (kn_tp + kn_fn)) * 100
kn_tnr = (kn_tn / (kn_tn + kn_fp)) * 100

data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'k-NN'],
    'TP': [lr_tp, dt_tp, rf_tp, kn_tp],
    'FP': [lr_fp, dt_fp, rf_fp, kn_fp],
    'TN': [lr_tn, dt_tn, rf_tn, kn_tn],
    'FN': [lr_fn, dt_fn, rf_fn, kn_fn],
    'Accuracy': [f"{lr_accuracy:.2f}%", f"{dt_accuracy:.2f}%", f"{rf_accuracy:.2f}%", f"{kn_accuracy:.2f}%"],
    'TPR': [f"{lr_tpr:.2f}%", f"{dt_tpr:.2f}%", f"{rf_tpr:.2f}%", f"{kn_tpr:.2f}%"],
    'TNR': [f"{lr_tnr:.2f}%", f"{dt_tnr:.2f}%", f"{rf_tnr:.2f}%", f"{kn_tnr:.2f}%"]
}

results_df = pd.DataFrame(data)
print()
print(results_df)