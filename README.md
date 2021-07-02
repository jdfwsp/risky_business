# 🐍🚨 Risky Business ⚠️🐍
 
![Credit Risk](Images/credit-risk.jpg)

## Background

Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked that you help them predict credit risk with machine learning techniques.

In this assignment I will build and evaluate several machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so I will need to employ different techniques for training and evaluating models with imbalanced classes. I will use the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](https://github.com/jdfwsp/risky_business/blob/main/Code/credit_risk_resampling.ipynb)
2. [Ensemble Learning](https://github.com/jdfwsp/risky_business/blob/main/Code/credit_risk_ensemble.ipynb)

### Procedure

#### Resampling

Imports:
```
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
```
To begin:

1. Read the CSV into a DataFrame.
```
file_path = Path('Resources/lending_data.csv')
df = pd.read_csv(file_path)
df = pd.get_dummies(df, columns=['homeowner'])
```
2. Split the data into Training and Testing sets.
```
X = df.drop(columns='loan_status')
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y)
```
3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.
```
data_scaler = StandardScaler()
data_scaler.fit(X_train)
X_train = data_scaler.transform(X_train)
X_test = data_scaler.transform(X_test)
```

4. Use the provided code to run a Simple Logistic Regression:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.
```
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
print(classification_report_imbalanced(y_test, y_pred))
```

Next you will:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.
```
ros = RandomOverSampler(random_state=1)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

smote = SMOTE(random_state=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```
2. Undersample the data using the `Cluster Centroids` algorithm.
```
cc = ClusterCentroids(random_state=1)
X_train_cc, y_train_cc = cc.fit_resample(X_train, y_train)
```
3. Over- and undersample using a combination `SMOTEENN` algorithm.
```
sm = SMOTEENN(random_state=1)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
```

For each of the above, you will need to:

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
```
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
2. Calculate the `balanced accuracy score` from `sklearn.metrics`.
```
balanced_accuracy_score(y_test, y_pred)
```
3. Display the `confusion matrix` from `sklearn.metrics`.
```
confusion_matrix(y_test, y_pred)
```
4. Print the `imbalanced classification report` from `imblearn.metrics`.
```
print(classification_report_imbalanced(y_test, y_pred))
```

Use the above to answer the following questions:

* Which model had the best balanced accuracy score?
> SMOTE Oversampling
* Which model had the best recall score?
> All models had avg/total recall score of 0.99
* Which model had the best geometric mean score?
> Every model had a geometric mean of 0.99
#### Ensemble Learning

In this section, I will train and compare two different ensemble classifiers to predict loan risk and evaluate each model. I will use the [Balanced Random Forest Classifier](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html#imblearn-ensemble-balancedrandomforestclassifier) and the [Easy Ensemble Classifier](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html#imblearn-ensemble-easyensembleclassifier).

Imports:
```
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
```
To begin:

1. Read the data into a DataFrame using the provided starter code.
```
file_path = Path('Resources/LoanStats_2019Q1.csv')
df = pd.read_csv(file_path)
dummify = df.select_dtypes(exclude='number').drop(columns='loan_status')
df = pd.get_dummies(df, columns=list(dummify.columns), drop_first=True)
```
2. Split the data into training and testing sets.
```
X = df.drop(columns='loan_status')
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y)
```
3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.
```
data_scaler = StandardScaler()
data_scaler.fit(X_train)
X_train = data_scaler.transform(X_train)
X_test = data_scaler.transform(X_test)
```
Then, complete the following steps for each model:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.
```
clf = BalancedRandomForestClassifier(n_estimators = 100,random_state=1)
clf = clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
```
2. Calculate the balanced accuracy score from `sklearn.metrics`.
```
balanced_accuracy_score(y_test, pred_y)
```
3. Display the confusion matrix from `sklearn.metrics`.
```
confusion_matrix(y_test, pred_y)
```
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.
```
print(classification_report_imbalanced(y_test, pred_y))
```
5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.
```
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = [df.columns[i] for i in indices]
```
Use the above to answer the following questions:

* Which model had the best balanced accuracy score?
> Easy Ensemble Classifier
* Which model had the best recall score?
> Easy Ensemble Classifier
* Which model had the best geometric mean score?
> Easy Ensemble Classifier
* What are the top three features?
> 'total_pymnt_inv', 'total_rec_prncp', 'out_prncp_inv'
- - -


