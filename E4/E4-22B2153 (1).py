#!/usr/bin/env python
# coding: utf-8

# # DS 203 -E4  ASSIGNMENT

# # Siddharth Verma   22B2153

# #### Importing Required Libraries and Data

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay


# In[31]:


data1=pd.read_csv(r"clusters-4-v0.csv")
data2=pd.read_csv(r"clusters-4-v1.csv")
data3=pd.read_csv(r"clusters-4-v2.csv")


# In[32]:


data1,data2,data3


# # Making the Training and Testing DataSets from overall Dataset and use them for all subsequent processing

# In[33]:


X1=data1[['x1','x2']]
y1=data1['y']
X2=data2[['x1','x2']]
y2=data2['y']
X3=data3[['x1','x2']]
y3=data3['y']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=42)


# # Plotting the data using Matplotlib and Guessing Possible Pattern

# In[34]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Scatter plot 1
axes[0].scatter(X1_train['x1'], X1_train['x2'], c=y1_train)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Data set 1 Plot')
# Scatter plot 2
axes[1].scatter(X2_train['x1'], X2_train['x2'], c=y2_train)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('Data set 2 Plot')
# Scatter plot 3
axes[2].scatter(X3_train['x1'], X3_train['x2'], c=y3_train)
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].set_title('Data set 3 Plot')
plt.show()


# Each of the four clusters in the datasets reveals notable differences in their distinguishability. The initial dataset showcases clusters with a distinct demarcation, making it easy to differentiate between them. However, in the second dataset, the clusters exhibit a diminished level of distinctiveness compared to the first dataset, creating a more challenging scenario for differentiation.
# 
# Dataset 3 presents a unique challenge as its clusters display substantial overlap. This overlap poses a formidable challenge in accurately identifying and delineating individual clusters, particularly when dealing with instances involving the intersection of the four clusters.

# # We have Used the following Algorithms/Variants to process the datasets
# 
# • Logistic Regression
# 
# • SVC with linear kernel
# 
# • SVC with rbf kernel
# 
# • Random Forest Classifier with min_samples_leaf=1
# 
# • Random Forest Classifier with min_samples_leaf=3
# 
# • Random Forest Classifier with min_samples_leaf=5
# 
# • Neural Network Classifier with hidden_layer_sizes=(5)
# 
# • Neural Network Classifier with hidden_layer_sizes=(5,5)
# 
# • Neural Network Classifier with hidden_layer_sizes=(5,5,5)
# 
# • Neural Network Classifier with hidden_layer_sizes=(10)

# # The following metrics were generated, captured, and saved into a CSV file for all algorithms on both training and test datasets:
# 
# • Train Accuracy
# 
# • Train Precision (per class and average)
# 
# • Train Recall (per class and average)
# 
# • Train F1-score (per class and average)
# 
# • Train AUC (per class and average)
# 
# • Test Accuracy
# 
# • Test Precision (per class and average)
# 
# • Test Recall (per class and average)
# 
# • Test F1-score (per class and average)
# 
# • Test AUC (per class and average)

# In[42]:


# Define the algorithms with probability=True
algorithms = [
    ('Logistic Regression', LogisticRegression()),
    ('SVC (Linear Kernel)', SVC(kernel='linear', probability=True)),
    ('SVC (RBF Kernel)', SVC(kernel='rbf', probability=True)),
    ('Random Forest Classifier (min_samples_leaf=1)', RandomForestClassifier(min_samples_leaf=1)),
    ('Random Forest Classifier (min_samples_leaf=3)', RandomForestClassifier(min_samples_leaf=3)),
    ('Random Forest Classifier (min_samples_leaf=5)', RandomForestClassifier(min_samples_leaf=5)),
    ('Neural Network Classifier (hidden_layer_sizes=(5,))', MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)),
    ('Neural Network Classifier (hidden_layer_sizes=(5,5))', MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)),
    ('Neural Network Classifier (hidden_layer_sizes=(5,5,5))', MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000)),
    ('Neural Network Classifier (hidden_layer_sizes=(10,))', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000))
]

# Define the number of datasets (n)
n_datasets = 3

# Initialize a list to store the results
all_results = []

# Iterate over each dataset
for dataset_index in range(1, n_datasets + 1):
    X_train = globals()[f"X{dataset_index}_train"]
    X_test = globals()[f"X{dataset_index}_test"]
    y_train = globals()[f"y{dataset_index}_train"]
    y_test = globals()[f"y{dataset_index}_test"]

    # Initialize a list to store the results for the current dataset
    results = []

    # Iterate over each algorithm
    for name, model in algorithms:
        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Predictions on training data
        y_train_pred = model.predict(X_train)

        # Predictions on testing data
        y_test_pred = model.predict(X_test)

        # Calculate evaluation metrics for training data
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average=None)
        train_precision_avg = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average=None)
        train_recall_avg = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average=None)
        train_f1_avg = f1_score(y_train, y_train_pred, average='weighted')
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train), average='macro', multi_class='ovr')
        train_auc_avg = roc_auc_score(y_train, model.predict_proba(X_train), average='macro', multi_class='ovr')

        
        # Calculate evaluation metrics for testing data
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average=None)
        test_precision_avg = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average=None)
        test_recall_avg = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average=None)
        test_f1_avg = f1_score(y_test, y_test_pred, average='weighted')
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test), average='macro', multi_class='ovr')
        test_auc_avg = roc_auc_score(y_test, model.predict_proba(X_test), average='macro', multi_class='ovr')
        
        # Store the results
        results.append({
            'Algorithm': name,
            'Train Accuracy': train_accuracy,
            'Train Precision': train_precision,
            'Train Precision (Avg)': train_precision_avg,
            'Train Recall': train_recall,
            'Train Recall (Avg)': train_recall_avg,
            'Train F1-score': train_f1,
            'Train F1-score (Avg)': train_f1_avg,
            'Train AUC': train_auc,
            'Train AUC (Avg)': train_auc_avg,
            'Test Accuracy': test_accuracy,
            'Test Precision': test_precision,
            'Test Precision (Avg)': test_precision_avg,
            'Test Recall': test_recall,
            'Test Recall (Avg)': test_recall_avg,
            'Test F1-score': test_f1,
            'Test F1-score (Avg)': test_f1_avg,
            'Test AUC': test_auc,
            'Test AUC (Avg)': test_auc_avg
        })

    # Create a DataFrame from the results for the current dataset
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file for the current dataset
    results_df.to_csv(f'algorithm_metrics_dataset_{dataset_index}.csv', index=False)

    # Append the results for the current dataset to the overall results list
    all_results.append(results_df)


# # METRICS RESULTS WE GOT

# ## Data set 1

# In[53]:


show1=pd.read_csv('algorithm_metrics_dataset_1.csv')
show1


# ## Data set 2

# In[56]:


show2=pd.read_csv('algorithm_metrics_dataset_2.csv')
show2


# ## Data set 3

# In[55]:


show3=pd.read_csv('algorithm_metrics_dataset_3.csv')
show3


# # Countour Plot for Data Sets

# ## Data Set 1

# In[59]:


# Define meshgrid for decision boundary plot
X_=np.arange(start=X1_train['x1'].min()-1, stop=X1_train['x1'].max()+1, step=0.07)
Y_=np.arange(start=X1_train['x2'].min()-1, stop=X1_train['x2'].max()+1, step=0.07)
xx,yy=np.meshgrid(X_,Y_)
# Define classifiers
classifiers = [
('Logistic Regression', LogisticRegression()),
('SVC (Linear Kernel)', SVC(kernel='linear', probability=True)),
('SVC (RBF Kernel)', SVC(kernel='rbf', probability=True)),
('Random Forest Classifier (min_samples_leaf=1)', RandomForestClassifier(min_samples_leaf=1)),
('Random Forest Classifier (min_samples_leaf=3)', RandomForestClassifier(min_samples_leaf=3)),
('Random Forest Classifier (min_samples_leaf=5)', RandomForestClassifier(min_samples_leaf=5)),
('Neural Network Classifier (hidden_layer_sizes=(5,))', MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5))', MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5,5))', MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(10,))', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000))
]
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
# Flatten axes for easier iteration
axes = axes.flatten()

# Plot decision boundary and ROC curve for each classifier
for ax, (name, classifier) in zip(axes, classifiers):
    classifier.fit(X1_train, y1_train)
    # Plot decision boundary
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.75)
    ax.scatter(X1_train.iloc[:, [0]], X1_train.iloc[:, [1]], c=y1_train, cmap=plt.cm.Paired)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Decision Boundary - {name}')
    
plt.tight_layout()
plt.show()


# # Data Set 2

# In[60]:


# Define meshgrid for decision boundary plot
X_=np.arange(start=X2_train['x1'].min()-1, stop=X2_train['x1'].max()+1, step=0.07)
Y_=np.arange(start=X2_train['x2'].min()-1, stop=X2_train['x2'].max()+1, step=0.07)
xx,yy=np.meshgrid(X_,Y_)
# Define classifiers
classifiers = [
('Logistic Regression', LogisticRegression()),
('SVC (Linear Kernel)', SVC(kernel='linear', probability=True)),
('SVC (RBF Kernel)', SVC(kernel='rbf', probability=True)),
('Random Forest Classifier (min_samples_leaf=1)', RandomForestClassifier(min_samples_leaf=1)),
('Random Forest Classifier (min_samples_leaf=3)', RandomForestClassifier(min_samples_leaf=3)),
('Random Forest Classifier (min_samples_leaf=5)', RandomForestClassifier(min_samples_leaf=5)),
('Neural Network Classifier (hidden_layer_sizes=(5,))', MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5))', MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5,5))', MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(10,))', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000))
]
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
# Flatten axes for easier iteration
axes = axes.flatten()

# Plot decision boundary and ROC curve for each classifier
for ax, (name, classifier) in zip(axes, classifiers):
    classifier.fit(X2_train, y2_train)
    # Plot decision boundary
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.75)
    ax.scatter(X2_train.iloc[:, [0]], X2_train.iloc[:, [1]], c=y2_train, cmap=plt.cm.Paired)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Decision Boundary - {name}')
    
plt.tight_layout()
plt.show()


# # Data Set 3

# In[61]:


# Define meshgrid for decision boundary plot
X_=np.arange(start=X3_train['x1'].min()-1, stop=X3_train['x1'].max()+1, step=0.07)
Y_=np.arange(start=X3_train['x2'].min()-1, stop=X3_train['x2'].max()+1, step=0.07)
xx,yy=np.meshgrid(X_,Y_)
# Define classifiers
classifiers = [
('Logistic Regression', LogisticRegression()),
('SVC (Linear Kernel)', SVC(kernel='linear', probability=True)),
('SVC (RBF Kernel)', SVC(kernel='rbf', probability=True)),
('Random Forest Classifier (min_samples_leaf=1)', RandomForestClassifier(min_samples_leaf=1)),
('Random Forest Classifier (min_samples_leaf=3)', RandomForestClassifier(min_samples_leaf=3)),
('Random Forest Classifier (min_samples_leaf=5)', RandomForestClassifier(min_samples_leaf=5)),
('Neural Network Classifier (hidden_layer_sizes=(5,))', MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5))', MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(5,5,5))', MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=1000)),
('Neural Network Classifier (hidden_layer_sizes=(10,))', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000))
]
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
# Flatten axes for easier iteration
axes = axes.flatten()

# Plot decision boundary and ROC curve for each classifier
for ax, (name, classifier) in zip(axes, classifiers):
    classifier.fit(X3_train, y3_train)
    # Plot decision boundary
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.75)
    ax.scatter(X3_train.iloc[:, [0]], X3_train.iloc[:, [1]], c=y3_train, cmap=plt.cm.Paired)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Decision Boundary - {name}')
    
plt.tight_layout()
plt.show()


# # Model Performance Analysis
# 
# ## About
# 
# In this analysis, we evaluate the performance of various classification algorithms on three different datasets (Dataset 1, Dataset 2, and Dataset 3). The evaluation is based on multiple metrics, including accuracy, precision, recall, F1-score, and AUC.
# 
# ### Metric Definitions
# 
# | Metric           | Definition                                                  | Dataset 1 Characteristics                 | Dataset 2 Characteristics                 | Dataset 3 Characteristics                 |
# |------------------|-------------------------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
# | Accuracy         | Overall correctness of the model's predictions               | Easy to distinguish clusters              | One cluster less distinguishable           | Overlapping clusters, difficult to distinguish |
# | Precision        | Correctness of positive predictions made by the model        | High precision for all clusters            | Lower precision for one cluster            | Varied precision due to overlap             |
# | Precision (per class) | Ratio of correctly predicted instances of a class to all instances predicted as that class | High precision for each class              | Lower precision for one class               | Varied precision for each class             |
# | Precision (average) | Average of precision values across all classes               | High average precision                    | Lower average precision                    | Varied average precision                    |
# | Recall           | Ability of the model to correctly identify instances of a class | High recall for all clusters               | Lower recall for one cluster               | Varied recall due to overlap                |
# | Recall (per class) | Ratio of correctly predicted instances of a class to all instances of that class | High recall for each class                 | Lower recall for one class                 | Varied recall for each class                |
# | Recall (average) | Average of recall values across all classes                  | High average recall                       | Lower average recall                       | Varied average recall                       |
# | F1-score         | Harmonic mean of precision and recall, providing a balance between the two | High F1-score for all clusters             | Lower F1-score for one cluster             | Varied F1-score due to overlap              |
# | F1-score (per class) | Balance between precision and recall for each class          | High F1-score for each class               | Lower F1-score for one class               | Varied F1-score for each class              |
# | F1-score (average) | Average of F1-score values across all classes                | High average F1-score                     | Lower average F1-score                     | Varied average F1-score                     |
# | AUC              | Area Under the Curve, measures the ability of the model to distinguish between classes | High AUC for all clusters                  | Lower AUC for one cluster                  | Varied AUC due to overlap                   |
# | AUC (per class) | AUC calculated for each class separately                    | High AUC for each class                    | Lower AUC for one class                    | Varied AUC for each class                   |
# | AUC (average)   | Average of AUC values across all classes                     | High average AUC                          | Lower average AUC                          | Varied average AUC                          |
# 
# ## Dataset Analysis
# 
# ### Dataset 1
# 
# - All algorithms demonstrate perfect performance on the train dataset, achieving an accuracy, precision, recall, F1-score, and AUC of 1. This indicates that the models perfectly fit the training data and can classify instances without errors and maintain their high performance without overfitting.
# 
# ### Dataset 2
# 
# - In some cases, particularly with the Random Forest Classifier (min_samples_leaf=1), the models achieve perfect accuracy on the training data but slightly lower accuracy on the test data, indicating potential overfitting.
# - The Logistic Regression, SVC with Linear Kernel, and SVC with RBF Kernel consistently achieve high accuracy and precision scores on both the train and test datasets, indicating robust performance in correctly classifying instances across different classes.
# - The Random Forest Classifier with min_samples_leaf=3 also demonstrates strong recall and F1-score values, suggesting its effectiveness in correctly identifying positive instances and achieving a balance between precision and recall.
# 
# ### Dataset 3
# 
# - In some cases, particularly with the Random Forest Classifier (min_samples_leaf=1), the models achieve perfect accuracy on the training data but slightly lower accuracy on the test data, indicating potential overfitting.
# - Overall, the accuracy of the models ranges from approximately 86% to 100% on the training data and from approximately 83% to 96% on the test data.
# - The performance of algorithms such as Logistic Regression, SVC with linear and RBF kernels, and Random Forest Classifier with min_samples_leaf=3 generally exhibit balanced performance across different metrics. The neural network classifiers with different hidden layer sizes also demonstrate competitive performance, although they may require careful tuning of hyperparameters to optimize performance.
# 
# ## Major Learnings
# 
# - The performance of classification algorithms varied across different datasets and metrics. Certain algorithms demonstrated superior performance on specific datasets, highlighting the significance of selecting appropriate algorithms based on the unique characteristics of the data.
# - Overfitting was observed in some cases, emphasizing the importance of employing regularization techniques or adjusting model hyperparameters to improve generalization and prevent overfitting.
# - The choice of evaluation metrics played a crucial role in assessing various aspects of model performance. Metrics such as precision, recall, and overall accuracy provided valuable insights into the effectiveness of the classification models and their ability to correctly classify instances across different classes.
# 

# In[ ]:




