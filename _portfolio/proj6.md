---
title: "Datascience project6"
date: 2018-08-14
header:
  teaser: /images/img6.jpg
---

#Sample project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
import matplotlib.gridspec as gridspec


from sklearn import preprocessing
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.metrics import confusion_matrix, precision_recall_curve,auc,roc_auc_score, \
            roc_curve,recall_score,classification_report,accuracy_score

import itertools
from itertools import cycle

import os
import zipfile
import glob
import warnings
warnings.filterwarnings('ignore')
```


```python
folder = 'C:/Users/Pradeep Krishnan/Desktop/Lending club/Loan Data'
extension = ".zip"

os.chdir(folder)

for item in os.listdir(folder):
    if item.endswith(extension):
        #print (os.path.abspath(item))
        filename = os.path.basename(item)
        #print (filename)
        zip_ref = zipfile.ZipFile(filename)
        zip_ref.extractall(folder)
        print (zip_ref)
        #zip_ref.close()
all_files=glob.glob(folder +'/*.csv')
frame = pd.DataFrame()
list_=[]
for files_ in all_files:
    df = pd.read_csv(files_,skiprows=[0])
    list_.append(df)
frame = pd.concat(list_)
```

    <zipfile.ZipFile filename='LoanStats3a.csv (2).zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats3b.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats3c.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats3d.csv (2).zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2016Q1.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2016Q2.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2016Q3.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2016Q4.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2017Q1.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2017Q2.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2017Q3.csv.zip' mode='r'>
    <zipfile.ZipFile filename='LoanStats_2017Q4.csv.zip' mode='r'>


    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,47) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,47,123,124,125,128,129,130,133) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,19,55) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,112) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    C:\Anaconda\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (0,123,124,125,128,129,130,133,139,140,141) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
      <th>debt_settlement_flag_date</th>
      <th>settlement_status</th>
      <th>settlement_date</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 145 columns</p>
</div>




```python
frame.shape
```




    (1765451, 145)




```python
frame=frame.dropna(thresh=0.99*len(frame), axis=1)
```


```python
frame.shape
```




    (1765451, 50)




```python
frame.columns
```




    Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'sub_grade', 'home_ownership', 'annual_inc',
           'verification_status', 'issue_d', 'loan_status', 'pymnt_plan',
           'purpose', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
           'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
           'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
           'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
           'last_credit_pull_d', 'collections_12_mths_ex_med', 'policy_code',
           'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths',
           'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens', 'hardship_flag',
           'disbursement_method', 'debt_settlement_flag'],
          dtype='object')




```python
loan_df = frame
```


```python
null_col_df = loan_df.isnull().sum(axis=0).sort_values(ascending=False).reset_index()
null_col_df.columns = ['column_name', 'missing_count']
null_col_df['missing_count'] = 100*((null_col_df['missing_count'])/len(loan_df))
y = np.arange(null_col_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(20,8))
ax.bar(y, null_col_df.missing_count.values, color='red')
ax.set_xticks(y)
ax.set_xticklabels(null_col_df.column_name.values, rotation = 'vertical')
ax.set_ylabel("% of missing values")
ax.set_title("Percentage of missing values in each column")
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_9_0.png)



```python
Lending_df = frame.dropna(axis=0,how='all')
```


```python
Lending_df.shape
```




    (1765426, 51)




```python
Lending_df.columns
```




    Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
           'installment', 'grade', 'sub_grade', 'home_ownership', 'annual_inc',
           'verification_status', 'issue_d', 'loan_status', 'pymnt_plan',
           'purpose', 'title', 'zip_code', 'addr_state', 'dti', 'delinq_2yrs',
           'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
           'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
           'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
           'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
           'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
           'last_credit_pull_d', 'collections_12_mths_ex_med', 'policy_code',
           'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths',
           'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens', 'hardship_flag',
           'disbursement_method', 'debt_settlement_flag'],
          dtype='object')




```python
loan_df = Lending_df
```


```python
null_col_df = loan_df.isnull().sum(axis=0).sort_values(ascending=False).reset_index()
null_col_df.columns = ['column_name', 'missing_count']
null_col_df['missing_count'] = 100*((null_col_df['missing_count'])/len(loan_df))
y = np.arange(null_col_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(20,8))
ax.bar(y, null_col_df.missing_count.values, color='red')
ax.set_xticks(y)
ax.set_xticklabels(null_col_df.column_name.values, rotation = 'vertical')
ax.set_ylabel("% of missing values")
ax.set_title("Percentage of missing values in each column")
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/sample13.jpg)



```python
loan_df.loan_status.value_counts()[:5]
```




    Current               807060
    Fully Paid            729182
    Charged Off           189063
    Late (31-120 days)     21015
    In Grace Period        11694
    Name: loan_status, dtype: int64




```python
plt.figure(figsize = (4,3))
g = loan_df.loan_status.value_counts()[:5].plot(kind='bar',alpha=.50)
g.set_title("Loan status - Target variable", fontsize=15)
g.set_ylabel("Number of loans", fontsize=10)
plt.xticks(rotation = 45)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_16_0.png)



```python
plt.figure(figsize = (6,5))

g = sns.boxplot(x='grade', y="loan_amnt", data=loan_df, order=['A','B','C','D','E','F','G'])
g.set_xticklabels(g.get_xticklabels())
g.set_xlabel("Loan grade", fontsize=12)
g.set_ylabel("Loan amount \$$", fontsize=12)
g.set_title("Loan amount vs loan grade", fontsize=15)

plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_17_0.png)



```python
plt.figure(figsize = (6,5))
a = frame.emp_length.value_counts(sort=False).plot(kind='bar',alpha=.60)
a.set_xlabel("Employment length", fontsize=12)
a.set_ylabel("Loan count", fontsize=12)
a.set_title("Loan count vs Employment length", fontsize=15)


plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_18_0.png)



```python
loan_df['int_rate'] = loan_df['int_rate'].str.rstrip('%').astype('float')
```


```python
plt.figure(figsize=(4,4))
loan_df = loan_df[(loan_df['loan_status'] == 'Fully Paid') | (loan_df.loan_status == 'Charged Off')]
g = sns.pointplot(x="grade", y="loan_amnt", hue="loan_status", data=loan_df, palette={"Fully Paid":"g", "Charged Off":"r"}, markers=["^","o"], linestyles=["-","--"], order=['A','B','C','D','E','F','G'])
g.set_xlabel("Loan grade", fontsize=12)
g.set_ylabel("Loan amount", fontsize=12)
g.set_title("Loan status", fontsize=12)
```




    Text(0.5,1,'Loan status')




![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_20_1.png)



```python
#loan_df = pd.read_csv("LoanStats_2016Q2.csv", skiprows=1, low_memory=False)
#loan_df.shape
```




    (97856, 145)




```python
loan_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
loan_df.columns.tolist()
```




    ['loan_amnt',
     'funded_amnt',
     'funded_amnt_inv',
     'term',
     'int_rate',
     'installment',
     'grade',
     'sub_grade',
     'home_ownership',
     'annual_inc',
     'verification_status',
     'issue_d',
     'loan_status',
     'pymnt_plan',
     'purpose',
     'zip_code',
     'addr_state',
     'dti',
     'delinq_2yrs',
     'earliest_cr_line',
     'inq_last_6mths',
     'open_acc',
     'pub_rec',
     'revol_bal',
     'revol_util',
     'total_acc',
     'initial_list_status',
     'out_prncp',
     'out_prncp_inv',
     'total_pymnt',
     'total_pymnt_inv',
     'total_rec_prncp',
     'total_rec_int',
     'total_rec_late_fee',
     'recoveries',
     'collection_recovery_fee',
     'last_pymnt_d',
     'last_pymnt_amnt',
     'last_credit_pull_d',
     'collections_12_mths_ex_med',
     'policy_code',
     'application_type',
     'acc_now_delinq',
     'chargeoff_within_12_mths',
     'delinq_amnt',
     'pub_rec_bankruptcies',
     'tax_liens',
     'hardship_flag',
     'disbursement_method',
     'debt_settlement_flag']



## Data Cleaning


```python
null_col_df = loan_df.isnull().sum(axis=0).sort_values(ascending=False).reset_index()
null_col_df.columns = ['column_name', 'missing_count']
null_col_df['missing_count'] = 100*((null_col_df['missing_count'])/len(loan_df))
y = np.arange(null_col_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(20,8))
ax.bar(y, null_col_df.missing_count.values, color='red')
ax.set_xticks(y)
ax.set_xticklabels(null_col_df.column_name.values, rotation = 'vertical')
ax.set_ylabel("% of missing values")
ax.set_title("Percentage of missing values in each column")
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_25_0.png)


#### We note that there are a large number of features missing more than 20% of values. Removing those features with more than 20% missing values.  


```python
null_col_df = loan_df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(loan_df))
loan_df.drop(null_col_df[null_col_df>0.1].index, axis=1, inplace = True)
loan_df.shape
```




    (1765451, 51)



## Taking a look at missing values in rows


```python
missing_row = loan_df.isnull().sum(axis=1).value_counts().reset_index()
missing_row.columns = ['number_of_missing_values', 'total_number_of_rows']
fig, ax = plt.subplots(figsize=(12,4))
ax.bar(missing_row['number_of_missing_values'], missing_row['total_number_of_rows'], color='red')
ax.set_title("Number of missing values in each row")
ax.set_ylabel("Number of rows")
ax.set_xticks(missing_row['number_of_missing_values'])
ax.set_xlabel("Number of missing values in each row")
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_29_0.png)



```python
#dropping all rows with any NA values
loan_df = loan_df.dropna()
loan_df.shape
```




    (915217, 50)




```python
#checking if we have any null values in dataframe
loan_df.isnull().values.any()
```




    False




```python
loan_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>...</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_amnt</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>9.152170e+05</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>...</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.0</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
      <td>915217.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14334.070035</td>
      <td>14321.796443</td>
      <td>14297.332242</td>
      <td>13.424609</td>
      <td>436.956948</td>
      <td>7.548250e+04</td>
      <td>17.974751</td>
      <td>0.310817</td>
      <td>0.704676</td>
      <td>11.555703</td>
      <td>...</td>
      <td>212.883937</td>
      <td>34.496449</td>
      <td>5825.788887</td>
      <td>0.015195</td>
      <td>1.0</td>
      <td>0.005035</td>
      <td>0.008744</td>
      <td>14.149030</td>
      <td>0.133061</td>
      <td>0.048028</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8540.206920</td>
      <td>8534.164848</td>
      <td>8534.736229</td>
      <td>4.654984</td>
      <td>256.906163</td>
      <td>6.523625e+04</td>
      <td>9.190019</td>
      <td>0.865045</td>
      <td>0.976833</td>
      <td>5.361516</td>
      <td>...</td>
      <td>859.081572</td>
      <td>149.061660</td>
      <td>7236.187869</td>
      <td>0.139003</td>
      <td>0.0</td>
      <td>0.077498</td>
      <td>0.107197</td>
      <td>745.982213</td>
      <td>0.376790</td>
      <td>0.382362</td>
    </tr>
    <tr>
      <th>min</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>5.320000</td>
      <td>4.930000</td>
      <td>1.000000e+02</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>7975.000000</td>
      <td>9.990000</td>
      <td>250.570000</td>
      <td>4.535700e+04</td>
      <td>11.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>446.230000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12000.000000</td>
      <td>12000.000000</td>
      <td>12000.000000</td>
      <td>12.990000</td>
      <td>376.450000</td>
      <td>6.500000e+04</td>
      <td>17.440000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2737.180000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>577.000000</td>
      <td>9.000000e+04</td>
      <td>23.750000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>14.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9004.460000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000000</td>
      <td>40000.000000</td>
      <td>40000.000000</td>
      <td>30.990000</td>
      <td>1714.540000</td>
      <td>9.550000e+06</td>
      <td>999.000000</td>
      <td>39.000000</td>
      <td>8.000000</td>
      <td>90.000000</td>
      <td>...</td>
      <td>39444.370000</td>
      <td>7002.190000</td>
      <td>42148.530000</td>
      <td>20.000000</td>
      <td>1.0</td>
      <td>14.000000</td>
      <td>10.000000</td>
      <td>94521.000000</td>
      <td>12.000000</td>
      <td>85.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>




```python
#Getting the datatype of the columns
loan_df.dtypes.value_counts()
```




    float64    30
    object     20
    dtype: int64




```python
#Look at only categorical columns
loan_df.select_dtypes(include=['object'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>home_ownership</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>purpose</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>earliest_cr_line</th>
      <th>revol_util</th>
      <th>initial_list_status</th>
      <th>last_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>application_type</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36 months</td>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>Jan-1985</td>
      <td>83.7%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60 months</td>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>car</td>
      <td>309xx</td>
      <td>GA</td>
      <td>Apr-1999</td>
      <td>9.4%</td>
      <td>f</td>
      <td>Apr-2013</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 months</td>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>small_business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Nov-2001</td>
      <td>98.5%</td>
      <td>f</td>
      <td>Jun-2014</td>
      <td>Jun-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36 months</td>
      <td>C</td>
      <td>C1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>917xx</td>
      <td>CA</td>
      <td>Feb-1996</td>
      <td>21%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Apr-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60 months</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>972xx</td>
      <td>OR</td>
      <td>Jan-1996</td>
      <td>53.9%</td>
      <td>f</td>
      <td>Jan-2017</td>
      <td>Jan-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>36 months</td>
      <td>A</td>
      <td>A4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>wedding</td>
      <td>852xx</td>
      <td>AZ</td>
      <td>Nov-2004</td>
      <td>28.3%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Feb-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60 months</td>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>280xx</td>
      <td>NC</td>
      <td>Jul-2005</td>
      <td>85.6%</td>
      <td>f</td>
      <td>May-2016</td>
      <td>Sep-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>36 months</td>
      <td>E</td>
      <td>E1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>car</td>
      <td>900xx</td>
      <td>CA</td>
      <td>Jan-2007</td>
      <td>87.5%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Dec-2014</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>60 months</td>
      <td>F</td>
      <td>F2</td>
      <td>OWN</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>small_business</td>
      <td>958xx</td>
      <td>CA</td>
      <td>Apr-2004</td>
      <td>32.6%</td>
      <td>f</td>
      <td>Apr-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60 months</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>774xx</td>
      <td>TX</td>
      <td>Sep-2004</td>
      <td>36.5%</td>
      <td>f</td>
      <td>Nov-2012</td>
      <td>Dec-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>60 months</td>
      <td>C</td>
      <td>C3</td>
      <td>OWN</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>853xx</td>
      <td>AZ</td>
      <td>Jan-1998</td>
      <td>20.6%</td>
      <td>f</td>
      <td>Jun-2013</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>11</th>
      <td>36 months</td>
      <td>B</td>
      <td>B5</td>
      <td>OWN</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>913xx</td>
      <td>CA</td>
      <td>Oct-1989</td>
      <td>67.1%</td>
      <td>f</td>
      <td>Sep-2013</td>
      <td>Aug-2013</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>12</th>
      <td>36 months</td>
      <td>C</td>
      <td>C1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>245xx</td>
      <td>VA</td>
      <td>Apr-2004</td>
      <td>91.7%</td>
      <td>f</td>
      <td>Jul-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>13</th>
      <td>36 months</td>
      <td>B</td>
      <td>B1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Jul-2003</td>
      <td>43.1%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Apr-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>36 months</td>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>951xx</td>
      <td>CA</td>
      <td>May-1991</td>
      <td>55.5%</td>
      <td>f</td>
      <td>Oct-2013</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>15</th>
      <td>36 months</td>
      <td>D</td>
      <td>D1</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>641xx</td>
      <td>MO</td>
      <td>Sep-2007</td>
      <td>81.5%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>16</th>
      <td>36 months</td>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>home_improvement</td>
      <td>921xx</td>
      <td>CA</td>
      <td>Oct-1998</td>
      <td>70.2%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Sep-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17</th>
      <td>36 months</td>
      <td>A</td>
      <td>A1</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>major_purchase</td>
      <td>067xx</td>
      <td>CT</td>
      <td>Aug-1993</td>
      <td>16%</td>
      <td>f</td>
      <td>May-2013</td>
      <td>May-2014</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>18</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>MORTGAGE</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>medical</td>
      <td>890xx</td>
      <td>UT</td>
      <td>Oct-2003</td>
      <td>37.73%</td>
      <td>f</td>
      <td>Feb-2015</td>
      <td>Jul-2015</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>19</th>
      <td>36 months</td>
      <td>A</td>
      <td>A1</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>921xx</td>
      <td>CA</td>
      <td>Jan-2001</td>
      <td>23.1%</td>
      <td>f</td>
      <td>Jul-2012</td>
      <td>Feb-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>60 months</td>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>770xx</td>
      <td>TX</td>
      <td>Nov-1997</td>
      <td>85.6%</td>
      <td>f</td>
      <td>Aug-2015</td>
      <td>Jun-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21</th>
      <td>36 months</td>
      <td>B</td>
      <td>B4</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>335xx</td>
      <td>FL</td>
      <td>Feb-1983</td>
      <td>90.3%</td>
      <td>f</td>
      <td>Sep-2013</td>
      <td>Feb-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>OWN</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>799xx</td>
      <td>TX</td>
      <td>Jul-1985</td>
      <td>82.4%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Jan-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>23</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>917xx</td>
      <td>CA</td>
      <td>Apr-2003</td>
      <td>91.8%</td>
      <td>f</td>
      <td>Oct-2013</td>
      <td>Mar-2014</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>24</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>major_purchase</td>
      <td>900xx</td>
      <td>CA</td>
      <td>Jun-2001</td>
      <td>29.7%</td>
      <td>f</td>
      <td>Oct-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>25</th>
      <td>36 months</td>
      <td>B</td>
      <td>B1</td>
      <td>MORTGAGE</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>605xx</td>
      <td>IL</td>
      <td>Feb-2002</td>
      <td>93.9%</td>
      <td>f</td>
      <td>Sep-2012</td>
      <td>Sep-2012</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>27</th>
      <td>60 months</td>
      <td>D</td>
      <td>D2</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>150xx</td>
      <td>PA</td>
      <td>Oct-2003</td>
      <td>59.5%</td>
      <td>f</td>
      <td>Dec-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>28</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>326xx</td>
      <td>FL</td>
      <td>Aug-1984</td>
      <td>37.7%</td>
      <td>f</td>
      <td>Apr-2013</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>29</th>
      <td>36 months</td>
      <td>B</td>
      <td>B3</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>564xx</td>
      <td>MN</td>
      <td>Nov-2006</td>
      <td>59.1%</td>
      <td>f</td>
      <td>Dec-2014</td>
      <td>Jan-2015</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>30</th>
      <td>36 months</td>
      <td>A</td>
      <td>A3</td>
      <td>MORTGAGE</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>141xx</td>
      <td>NY</td>
      <td>Dec-1987</td>
      <td>86.9%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>118120</th>
      <td>36 months</td>
      <td>B</td>
      <td>B5</td>
      <td>MORTGAGE</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>923xx</td>
      <td>CA</td>
      <td>May-2004</td>
      <td>37.3%</td>
      <td>w</td>
      <td>Feb-2018</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118125</th>
      <td>36 months</td>
      <td>C</td>
      <td>C1</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>961xx</td>
      <td>CA</td>
      <td>Sep-2001</td>
      <td>61.8%</td>
      <td>w</td>
      <td>Feb-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118153</th>
      <td>36 months</td>
      <td>D</td>
      <td>D1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>major_purchase</td>
      <td>070xx</td>
      <td>NJ</td>
      <td>Nov-2003</td>
      <td>23.4%</td>
      <td>w</td>
      <td>Oct-2017</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118180</th>
      <td>36 months</td>
      <td>B</td>
      <td>B4</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>240xx</td>
      <td>VA</td>
      <td>Sep-2003</td>
      <td>31.3%</td>
      <td>w</td>
      <td>Feb-2018</td>
      <td>Jan-2018</td>
      <td>Joint App</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118194</th>
      <td>36 months</td>
      <td>A</td>
      <td>A5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>641xx</td>
      <td>MO</td>
      <td>Jul-1999</td>
      <td>25.2%</td>
      <td>w</td>
      <td>Nov-2017</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118195</th>
      <td>60 months</td>
      <td>C</td>
      <td>C3</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>891xx</td>
      <td>NV</td>
      <td>May-2006</td>
      <td>27.7%</td>
      <td>w</td>
      <td>Mar-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118196</th>
      <td>36 months</td>
      <td>B</td>
      <td>B4</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>452xx</td>
      <td>OH</td>
      <td>Apr-1996</td>
      <td>48.8%</td>
      <td>f</td>
      <td>Nov-2017</td>
      <td>Jan-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118198</th>
      <td>36 months</td>
      <td>B</td>
      <td>B1</td>
      <td>MORTGAGE</td>
      <td>Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>home_improvement</td>
      <td>335xx</td>
      <td>FL</td>
      <td>Nov-2001</td>
      <td>18.6%</td>
      <td>w</td>
      <td>Mar-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118250</th>
      <td>36 months</td>
      <td>C</td>
      <td>C4</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>115xx</td>
      <td>NY</td>
      <td>Feb-2000</td>
      <td>34.2%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118251</th>
      <td>36 months</td>
      <td>C</td>
      <td>C3</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>750xx</td>
      <td>TX</td>
      <td>Dec-2003</td>
      <td>10.4%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118256</th>
      <td>36 months</td>
      <td>D</td>
      <td>D2</td>
      <td>MORTGAGE</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>604xx</td>
      <td>IL</td>
      <td>Nov-1999</td>
      <td>57.3%</td>
      <td>f</td>
      <td>Jan-2018</td>
      <td>Jan-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118284</th>
      <td>36 months</td>
      <td>C</td>
      <td>C3</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>900xx</td>
      <td>CA</td>
      <td>Nov-2005</td>
      <td>18.1%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118354</th>
      <td>36 months</td>
      <td>G</td>
      <td>G3</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>920xx</td>
      <td>CA</td>
      <td>Oct-2003</td>
      <td>50.5%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118371</th>
      <td>36 months</td>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>928xx</td>
      <td>CA</td>
      <td>May-1994</td>
      <td>19.1%</td>
      <td>f</td>
      <td>Oct-2017</td>
      <td>Sep-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118375</th>
      <td>36 months</td>
      <td>B</td>
      <td>B5</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>car</td>
      <td>481xx</td>
      <td>MI</td>
      <td>Apr-1998</td>
      <td>58.9%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118397</th>
      <td>36 months</td>
      <td>E</td>
      <td>E2</td>
      <td>OWN</td>
      <td>Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>902xx</td>
      <td>CA</td>
      <td>Mar-1990</td>
      <td>90.7%</td>
      <td>f</td>
      <td>Mar-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118445</th>
      <td>36 months</td>
      <td>D</td>
      <td>D2</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>940xx</td>
      <td>CA</td>
      <td>May-2004</td>
      <td>54.1%</td>
      <td>f</td>
      <td>Oct-2017</td>
      <td>Sep-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118449</th>
      <td>36 months</td>
      <td>E</td>
      <td>E2</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>532xx</td>
      <td>WI</td>
      <td>Sep-1995</td>
      <td>55.1%</td>
      <td>f</td>
      <td>Nov-2017</td>
      <td>Dec-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118486</th>
      <td>36 months</td>
      <td>C</td>
      <td>C5</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>325xx</td>
      <td>FL</td>
      <td>Sep-2006</td>
      <td>65.1%</td>
      <td>f</td>
      <td>Dec-2017</td>
      <td>Jan-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118510</th>
      <td>36 months</td>
      <td>C</td>
      <td>C1</td>
      <td>MORTGAGE</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Jul-1999</td>
      <td>55.6%</td>
      <td>f</td>
      <td>Mar-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118523</th>
      <td>36 months</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Oct-1995</td>
      <td>40%</td>
      <td>f</td>
      <td>Nov-2017</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118524</th>
      <td>36 months</td>
      <td>D</td>
      <td>D3</td>
      <td>MORTGAGE</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>home_improvement</td>
      <td>366xx</td>
      <td>AL</td>
      <td>May-2004</td>
      <td>36.4%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118545</th>
      <td>36 months</td>
      <td>E</td>
      <td>E5</td>
      <td>OWN</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>060xx</td>
      <td>CT</td>
      <td>Feb-1995</td>
      <td>70%</td>
      <td>f</td>
      <td>Oct-2017</td>
      <td>Sep-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118568</th>
      <td>36 months</td>
      <td>E</td>
      <td>E1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>912xx</td>
      <td>CA</td>
      <td>Jan-2004</td>
      <td>66.7%</td>
      <td>f</td>
      <td>Nov-2017</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118575</th>
      <td>60 months</td>
      <td>D</td>
      <td>D5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>117xx</td>
      <td>NY</td>
      <td>Feb-2004</td>
      <td>61.5%</td>
      <td>f</td>
      <td>Nov-2017</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118576</th>
      <td>36 months</td>
      <td>D</td>
      <td>D5</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>601xx</td>
      <td>IL</td>
      <td>Mar-2005</td>
      <td>20.7%</td>
      <td>f</td>
      <td>Jan-2018</td>
      <td>Jan-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118579</th>
      <td>36 months</td>
      <td>E</td>
      <td>E5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>small_business</td>
      <td>840xx</td>
      <td>UT</td>
      <td>Oct-2006</td>
      <td>51%</td>
      <td>f</td>
      <td>Oct-2017</td>
      <td>Sep-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118593</th>
      <td>36 months</td>
      <td>E</td>
      <td>E4</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>731xx</td>
      <td>OK</td>
      <td>Jan-2006</td>
      <td>96.5%</td>
      <td>f</td>
      <td>Mar-2018</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118620</th>
      <td>60 months</td>
      <td>F</td>
      <td>F5</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>852xx</td>
      <td>AZ</td>
      <td>Nov-2003</td>
      <td>17.2%</td>
      <td>f</td>
      <td>Feb-2018</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>118630</th>
      <td>60 months</td>
      <td>F</td>
      <td>F4</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Oct-2017</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>841xx</td>
      <td>UT</td>
      <td>Aug-1981</td>
      <td>11.6%</td>
      <td>f</td>
      <td>Oct-2017</td>
      <td>Feb-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>915217 rows × 20 columns</p>
</div>




```python
#Loan Status is the target variable. Exploring it further...
#percentage of each loan status
loan_df.loan_status.value_counts().plot(kind='bar',alpha=.50)

plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_35_0.png)


## Looking at columns to see what is needed and what can be removed

## Data cleaning in Float columns


```python

#Removing funded_amnt, funded_amnt_inv,  
#check delinq_2yrs -- seems ok
#check out_prncp and out_prncp_inv as its about future
#check total_pymnt and total_pymnt_inv as its about future
#check total_rec_prncp, total_rec_int, total_rec_late_fee as its about future
#check recoveries, collection_recovery_fee, last_pymnt_d, last_pymnt_amnt
#as its about future
#check total_rec_late_fee, recoveries, collection_recovery_fee as all appear 0
#collections_12_mths_ex_med, policy_code, acc_now_delinq,  
#check these:
#tot_coll_amt - sparse?
#open_acc_6m - ??
#open_act_il
#open_il_12m
#mths_since_rcnt_il
#open_rv_12m
#open_rv_24m
#inq_fi
#total_cu_tl
#inq_last_12m
#acc_open_past_24mths
#delinq_amnt
#num_accts_ever_120_pd
#num_tl_120dpd_2m
#num_tl_30dpd
#num_tl_90g_dpd_24m
#pub_rec_bankruptcies
#tax_liens

#Removing



```

## Data cleaning in Object columns


```python
#Removing sub_grade, int_rate, emp_title, issue_d
#Removing zip_code,
```


```python
#Loading the data dictionary provided by Lending Club
datadict_df = pd.read_excel("LCDataDictionary.xlsx")
datadict_df.columns = ['name','description']
datadict_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>acc_now_delinq</td>
      <td>The number of accounts on which the borrower i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>acc_open_past_24mths</td>
      <td>Number of trades opened in past 24 months.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>all_util</td>
      <td>Balance to credit limit on all trades</td>
    </tr>
    <tr>
      <th>4</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by th...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>annual_inc_joint</td>
      <td>The combined self-reported annual income provi...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual ap...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>avg_cur_bal</td>
      <td>Average current balance of all accounts</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bc_open_to_buy</td>
      <td>Total open to buy on revolving bankcards.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bc_util</td>
      <td>Ratio of total current balance to high credit/...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>chargeoff_within_12_mths</td>
      <td>Number of charge-offs within 12 months</td>
    </tr>
    <tr>
      <th>11</th>
      <td>collection_recovery_fee</td>
      <td>post charge off collection fee</td>
    </tr>
    <tr>
      <th>12</th>
      <td>collections_12_mths_ex_med</td>
      <td>Number of collections in 12 months excluding m...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>delinq_2yrs</td>
      <td>The number of 30+ days past-due incidences of ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>delinq_amnt</td>
      <td>The past-due amount owed for the accounts on w...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>desc</td>
      <td>Loan description provided by the borrower</td>
    </tr>
    <tr>
      <th>16</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti_joint</td>
      <td>A ratio calculated using the co-borrowers' tot...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported cre...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values ar...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when ap...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>fico_range_high</td>
      <td>The upper boundary range the borrower’s FICO a...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>fico_range_low</td>
      <td>The lower boundary range the borrower’s FICO a...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>funded_amnt</td>
      <td>The total amount committed to that loan at tha...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>funded_amnt_inv</td>
      <td>The total amount committed by investors for th...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>26</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borr...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>id</td>
      <td>A unique LC assigned ID for the loan listing.</td>
    </tr>
    <tr>
      <th>28</th>
      <td>il_util</td>
      <td>Ratio of total current balance to high credit/...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possib...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>sec_app_open_act_il</td>
      <td>Number of currently active installment trades...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>sec_app_num_rev_accts</td>
      <td>Number of revolving accounts at time of appli...</td>
    </tr>
    <tr>
      <th>125</th>
      <td>sec_app_chargeoff_within_12_mths</td>
      <td>Number of charge-offs within last 12 months a...</td>
    </tr>
    <tr>
      <th>126</th>
      <td>sec_app_collections_12_mths_ex_med</td>
      <td>Number of collections within last 12 months e...</td>
    </tr>
    <tr>
      <th>127</th>
      <td>sec_app_mths_since_last_major_derog</td>
      <td>Months since most recent 90-day or worse rati...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>hardship_flag</td>
      <td>Flags whether or not the borrower is on a hard...</td>
    </tr>
    <tr>
      <th>129</th>
      <td>hardship_type</td>
      <td>Describes the hardship plan offering</td>
    </tr>
    <tr>
      <th>130</th>
      <td>hardship_reason</td>
      <td>Describes the reason the hardship plan was off...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>hardship_status</td>
      <td>Describes if the hardship plan is active, pend...</td>
    </tr>
    <tr>
      <th>132</th>
      <td>deferral_term</td>
      <td>Amount of months that the borrower is expected...</td>
    </tr>
    <tr>
      <th>133</th>
      <td>hardship_amount</td>
      <td>The interest payment that the borrower has com...</td>
    </tr>
    <tr>
      <th>134</th>
      <td>hardship_start_date</td>
      <td>The start date of the hardship plan period</td>
    </tr>
    <tr>
      <th>135</th>
      <td>hardship_end_date</td>
      <td>The end date of the hardship plan period</td>
    </tr>
    <tr>
      <th>136</th>
      <td>payment_plan_start_date</td>
      <td>The day the first hardship plan payment is due...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>hardship_length</td>
      <td>The number of months the borrower will make sm...</td>
    </tr>
    <tr>
      <th>138</th>
      <td>hardship_dpd</td>
      <td>Account days past due as of the hardship plan ...</td>
    </tr>
    <tr>
      <th>139</th>
      <td>hardship_loan_status</td>
      <td>Loan Status as of the hardship plan start date</td>
    </tr>
    <tr>
      <th>140</th>
      <td>orig_projected_additional_accrued_interest</td>
      <td>The original projected additional interest amo...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>hardship_payoff_balance_amount</td>
      <td>The payoff balance amount as of the hardship p...</td>
    </tr>
    <tr>
      <th>142</th>
      <td>hardship_last_payment_amount</td>
      <td>The last payment amount as of the hardship pla...</td>
    </tr>
    <tr>
      <th>143</th>
      <td>disbursement_method</td>
      <td>The method by which the borrower receives thei...</td>
    </tr>
    <tr>
      <th>144</th>
      <td>debt_settlement_flag</td>
      <td>Flags whether or not the borrower, who has cha...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>debt_settlement_flag_date</td>
      <td>The most recent date that the Debt_Settlement_...</td>
    </tr>
    <tr>
      <th>146</th>
      <td>settlement_status</td>
      <td>The status of the borrower’s settlement plan. ...</td>
    </tr>
    <tr>
      <th>147</th>
      <td>settlement_date</td>
      <td>The date that the borrower agrees to the settl...</td>
    </tr>
    <tr>
      <th>148</th>
      <td>settlement_amount</td>
      <td>The loan amount that the borrower has agreed t...</td>
    </tr>
    <tr>
      <th>149</th>
      <td>settlement_percentage</td>
      <td>The settlement amount as a percentage of the p...</td>
    </tr>
    <tr>
      <th>150</th>
      <td>settlement_term</td>
      <td>The number of months that the borrower will be...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>152</th>
      <td>NaN</td>
      <td>* Employer Title replaces Employer Name for al...</td>
    </tr>
  </tbody>
</table>
<p>153 rows × 2 columns</p>
</div>




```python
cols_to_drop = ['debt_settlement_flag','funded_amnt_inv', 'delinq_2yrs', 'out_prncp', 'out_prncp_inv', 'total_pymnt','total_pymnt_inv',
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
                'last_pymnt_amnt', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med',
                'policy_code', 'acc_now_delinq', 'tot_coll_amt', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'mths_since_rcnt_il',
                'open_rv_12m', 'open_rv_24m', 'inq_fi', 'total_cu_tl','inq_last_12m', 'acc_open_past_24mths', 'delinq_amnt',
                'num_accts_ever_120_pd', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'pub_rec_bankruptcies','tax_liens',
               'issue_d','pymnt_plan', 'hardship_flag', 'zip_code', 'addr_state', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']



```


```python
#loan_df = loan_df.drop(cols_to_drop, axis=1)
```


```python
loan_df.shape
```




    (915217, 50)



## Visualizing object features


```python
loan_name_df = pd.DataFrame(loan_df.dtypes, columns = ['type'])
loan_name_df = loan_name_df.reset_index()
loan_name_df['name'] = loan_name_df['index']
loan_name_df = loan_name_df[['name','type']]
loan_name_df = loan_name_df[loan_name_df['type'] == 'object']

name_description_df = loan_name_df.merge(datadict_df, on = 'name', how = 'left')
name_description_df = name_description_df[1:]
pd.set_option('display.max_colwidth', -1)
name_description_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>type</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>grade</td>
      <td>object</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub_grade</td>
      <td>object</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>3</th>
      <td>home_ownership</td>
      <td>object</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>verification_status</td>
      <td>object</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>5</th>
      <td>issue_d</td>
      <td>object</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>6</th>
      <td>loan_status</td>
      <td>object</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pymnt_plan</td>
      <td>object</td>
      <td>Indicates if a payment plan has been put in place for the loan</td>
    </tr>
    <tr>
      <th>8</th>
      <td>purpose</td>
      <td>object</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>zip_code</td>
      <td>object</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>addr_state</td>
      <td>object</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>11</th>
      <td>earliest_cr_line</td>
      <td>object</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>12</th>
      <td>revol_util</td>
      <td>object</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>initial_list_status</td>
      <td>object</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>14</th>
      <td>last_pymnt_d</td>
      <td>object</td>
      <td>Last month payment was received</td>
    </tr>
    <tr>
      <th>15</th>
      <td>last_credit_pull_d</td>
      <td>object</td>
      <td>The most recent month LC pulled credit for this loan</td>
    </tr>
    <tr>
      <th>16</th>
      <td>application_type</td>
      <td>object</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>17</th>
      <td>hardship_flag</td>
      <td>object</td>
      <td>Flags whether or not the borrower is on a hardship plan</td>
    </tr>
    <tr>
      <th>18</th>
      <td>disbursement_method</td>
      <td>object</td>
      <td>The method by which the borrower receives their loan. Possible values are: CASH, DIRECT_PAY</td>
    </tr>
    <tr>
      <th>19</th>
      <td>debt_settlement_flag</td>
      <td>object</td>
      <td>Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company.</td>
    </tr>
  </tbody>
</table>
</div>




```python
object_df = loan_df.select_dtypes(include=['object']).head(10)
pd.set_option('display.max_colwidth', -1)
object_df.iloc[:10,]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>home_ownership</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>purpose</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>earliest_cr_line</th>
      <th>revol_util</th>
      <th>initial_list_status</th>
      <th>last_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>application_type</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36 months</td>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>Jan-1985</td>
      <td>83.7%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60 months</td>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>car</td>
      <td>309xx</td>
      <td>GA</td>
      <td>Apr-1999</td>
      <td>9.4%</td>
      <td>f</td>
      <td>Apr-2013</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 months</td>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>small_business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Nov-2001</td>
      <td>98.5%</td>
      <td>f</td>
      <td>Jun-2014</td>
      <td>Jun-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36 months</td>
      <td>C</td>
      <td>C1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>917xx</td>
      <td>CA</td>
      <td>Feb-1996</td>
      <td>21%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Apr-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60 months</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>972xx</td>
      <td>OR</td>
      <td>Jan-1996</td>
      <td>53.9%</td>
      <td>f</td>
      <td>Jan-2017</td>
      <td>Jan-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>36 months</td>
      <td>A</td>
      <td>A4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>wedding</td>
      <td>852xx</td>
      <td>AZ</td>
      <td>Nov-2004</td>
      <td>28.3%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Feb-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>60 months</td>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>280xx</td>
      <td>NC</td>
      <td>Jul-2005</td>
      <td>85.6%</td>
      <td>f</td>
      <td>May-2016</td>
      <td>Sep-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>36 months</td>
      <td>E</td>
      <td>E1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>car</td>
      <td>900xx</td>
      <td>CA</td>
      <td>Jan-2007</td>
      <td>87.5%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Dec-2014</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>60 months</td>
      <td>F</td>
      <td>F2</td>
      <td>OWN</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>small_business</td>
      <td>958xx</td>
      <td>CA</td>
      <td>Apr-2004</td>
      <td>32.6%</td>
      <td>f</td>
      <td>Apr-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60 months</td>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>774xx</td>
      <td>TX</td>
      <td>Sep-2004</td>
      <td>36.5%</td>
      <td>f</td>
      <td>Nov-2012</td>
      <td>Dec-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Looking at columns with less than 5 unique values
for c in loan_df.columns:
    if(loan_df[c].nunique()<5):
        print(loan_df[c].name)
        print(loan_df[c].value_counts())
        print()
```

    term
     36 months    694975
     60 months    220242
    Name: term, dtype: int64

    verification_status
    Source Verified    335096
    Verified           298820
    Not Verified       281301
    Name: verification_status, dtype: int64

    loan_status
    Fully Paid     728074
    Charged Off    187143
    Name: loan_status, dtype: int64

    pymnt_plan
    n    915217
    Name: pymnt_plan, dtype: int64

    initial_list_status
    w    468051
    f    447166
    Name: initial_list_status, dtype: int64

    out_prncp
    0.0    915217
    Name: out_prncp, dtype: int64

    out_prncp_inv
    0.0    915217
    Name: out_prncp_inv, dtype: int64

    policy_code
    1.0    915217
    Name: policy_code, dtype: int64

    application_type
    Individual    909192
    Joint App     6025  
    Name: application_type, dtype: int64

    hardship_flag
    N    915217
    Name: hardship_flag, dtype: int64

    disbursement_method
    Cash         913806
    DirectPay    1411  
    Name: disbursement_method, dtype: int64

    debt_settlement_flag
    N    898245
    Y    16972
    Name: debt_settlement_flag, dtype: int64



## Replacing values and setting right datatype


```python
#loan_df['term'] = loan_df['term'].str.rstrip(' months').astype('int')
#loan_df['int_rate'] = loan_df['int_rate'].str.rstrip('%').astype('float')
#loan_df['revol_util'] = loan_df['revol_util'].str.rstrip('%').astype('float')

#loan_df['emp_length'] = loan_df['emp_length'].replace(to_replace='[^0-9]+', value='', regex=True).astype('int')
```


```python
object_df = loan_df.select_dtypes(include=['object']).head(10)
pd.set_option('display.max_colwidth', -1)
object_df.iloc[:10,]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>home_ownership</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>purpose</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>earliest_cr_line</th>
      <th>revol_util</th>
      <th>initial_list_status</th>
      <th>last_pymnt_d</th>
      <th>last_credit_pull_d</th>
      <th>application_type</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>B2</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>credit_card</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>Jan-1985</td>
      <td>83.7%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Mar-2018</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>C4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>car</td>
      <td>309xx</td>
      <td>GA</td>
      <td>Apr-1999</td>
      <td>9.4%</td>
      <td>f</td>
      <td>Apr-2013</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>small_business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>Nov-2001</td>
      <td>98.5%</td>
      <td>f</td>
      <td>Jun-2014</td>
      <td>Jun-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C</td>
      <td>C1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>917xx</td>
      <td>CA</td>
      <td>Feb-1996</td>
      <td>21%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Apr-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>other</td>
      <td>972xx</td>
      <td>OR</td>
      <td>Jan-1996</td>
      <td>53.9%</td>
      <td>f</td>
      <td>Jan-2017</td>
      <td>Jan-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A</td>
      <td>A4</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>wedding</td>
      <td>852xx</td>
      <td>AZ</td>
      <td>Nov-2004</td>
      <td>28.3%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Feb-2017</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C</td>
      <td>C5</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>280xx</td>
      <td>NC</td>
      <td>Jul-2005</td>
      <td>85.6%</td>
      <td>f</td>
      <td>May-2016</td>
      <td>Sep-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>E</td>
      <td>E1</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>car</td>
      <td>900xx</td>
      <td>CA</td>
      <td>Jan-2007</td>
      <td>87.5%</td>
      <td>f</td>
      <td>Jan-2015</td>
      <td>Dec-2014</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>F</td>
      <td>F2</td>
      <td>OWN</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>small_business</td>
      <td>958xx</td>
      <td>CA</td>
      <td>Apr-2004</td>
      <td>32.6%</td>
      <td>f</td>
      <td>Apr-2012</td>
      <td>Oct-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B</td>
      <td>B5</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>774xx</td>
      <td>TX</td>
      <td>Sep-2004</td>
      <td>36.5%</td>
      <td>f</td>
      <td>Nov-2012</td>
      <td>Dec-2016</td>
      <td>Individual</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
loan_df = loan_df[(loan_df['loan_status'] == 'Fully Paid') | (loan_df.loan_status == 'Charged Off')]
loan_df['loan_status'].value_counts()
```




    Fully Paid     728074
    Charged Off    187143
    Name: loan_status, dtype: int64




```python
plt.figure(figsize = (6,5))

g = sns.boxplot(x='grade', y="loan_amnt", data=loan_df, order=['A','B','C','D','E','F','G'])
g.set_xticklabels(g.get_xticklabels())
g.set_xlabel("Loan grade", fontsize=12)
g.set_ylabel("Loan amount", fontsize=12)
g.set_title("Loan amount vs loan grade", fontsize=15)

plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_53_0.png)



```python
#loan_df.emp_length.value_counts
```


```python
plt.figure(figsize = (6,5))
a = frame.emp_length.value_counts().plot(kind='bar',alpha=.60)
a.set_xlabel("Employment length", fontsize=12)
a.set_ylabel("Loan count", fontsize=12)
a.set_title("Loan count vs Employment length", fontsize=15)


plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_55_0.png)



```python
emp_df = pd.DataFrame(frame.emp_length.value_counts()).reset_index()
emp_df.columns = ['emp_length', 'number_loans']

#null_col_df.columns = ['column_name', 'missing_count']
emp_df['number_loans'] = 100*((emp_df['number_loans'])/len(loan_df))
ax = sns.barplot(alpha=.60, x="emp_length", y="number_loans", data=emp_df)
ax.set_xlabel("years of employment", fontsize=12)
ax.set_ylabel("% of loans", fontsize=12)
ax.set_title("Employment Length vs Number of Loans", fontsize=12)
plt.show()

#y = np.arange(null_col_df.shape[0])
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_56_0.png)



```python
plt.figure(figsize = (4,3))
ax = sns.barplot(alpha=.60, x="grade", y="int_rate", data=loan_df, order=['A','B','C','D','E','F','G'])
ax.set_xlabel("Loan Grade", fontsize=12)
ax.set_ylabel("Interest Rate", fontsize=12)
ax.set_title("Loan Grade vs Interest Rate", fontsize=12)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_57_0.png)



```python
f, ax = plt.subplots(figsize=(8, 5))
loan_df['int_round'] = loan_df['int_rate'].round(0).astype(int)

g1 = sns.countplot(x="int_round",data=loan_df,
                   palette="Set2")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Interest Rate Distribuition", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_58_0.png)



```python
f, ax = plt.subplots(figsize=(12, 5))

g1 = sns.countplot(x="grade",data=loan_df,
                   palette="Set2",  order=['A','B','C','D','E','F','G'])
g1.set_xlabel("Loan grade", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Loan Grade Distribuition", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_59_0.png)



```python
f, ax = plt.subplots(figsize=(10, 6))
loan_amt_mean = loan_df.groupby('grade').agg([np.mean])['loan_amnt'].reset_index()

ax = sns.barplot(x="grade", y="mean", data=loan_amt_mean, order=['A','B','C','D','E','F','G'])
ax.set_ylabel("average loan amount", fontsize=12)

plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_60_0.png)



```python
f, ax = plt.subplots(figsize=(14, 8))
sns.violinplot(x='grade', y='int_rate', hue='loan_status', data=loan_df, order=['A','B','C','D','E','F','G'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27d2325a080>




![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_61_1.png)



```python
f, ax = plt.subplots(figsize=(6, 4))
loan_df = loan_df[(loan_df['loan_status'] == 'Fully Paid') | (loan_df.loan_status == 'Charged Off')]

sns.violinplot(x='term', y='funded_amnt', hue='loan_status', data=loan_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x285c4c8e6d8>




![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_62_1.png)



```python
f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x='grade', y='int_rate', hue='home_ownership', data=loan_df, order=['A','B','C','D','E','F','G'])
ax.set_ylabel("Interest Rate", fontsize=12)
ax.set_xlabel("Loan Grade", fontsize=12)
ax.set_title("Loan Grade vs Interest rate and Home ownership", fontsize = 15)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_63_0.png)



```python
f, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='emp_length', data=loan_df)
plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_64_0.png)



```python
plt.figure(figsize=(3,3))
loan_df = loan_df[loan_df['annual_inc'] < 175000]
g = sns.jointplot(x=loan_df.annual_inc, y=loan_df.funded_amnt, size=10, color=color[3])
#g.ylabel('funded amount', fontsize=12)
#g.xlabel('annual income', fontsize=12)
#plt.title("Loan amount vs interest rate", fontsize=15)
plt.show()
```


    <matplotlib.figure.Figure at 0x1b40dd54a20>



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_65_1.png)



```python
plt.figure(figsize=(6,6))

sns.pointplot(x="grade", y="int_rate", hue="loan_status", data=loan_df, palette={"Fully Paid":"g", "Charged Off":"r"}, markers=["^","o"], linestyles=["-","--"], order=['A','B','C','D','E','F','G'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27d25c87860>




![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_66_1.png)



```python
plt.figure(figsize=(6,6))
sns.pointplot(x="grade", y="loan_amnt", hue="loan_status", data=loan_df, palette={"Fully Paid":"g", "Charged Off":"r"}, markers=["^","o"], linestyles=["-","--"], order=['A','B','C','D','E','F','G'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27d24da7748>




![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_67_1.png)



```python
loan_df['loan_status'] = loan_df['loan_status'].map({'Fully Paid': 1, 'Charged Off':0})
loan_df['loan_status'] = loan_df['loan_status'].apply(lambda loan_status:0 if loan_status == 0 else 1)
float_df = loan_df.select_dtypes(exclude = ['object'])
float_df['loan_status'].value_counts()
```




    1    701176
    0    182513
    Name: loan_status, dtype: int64




```python
columns = float_df.iloc[:,1:47].columns

grid = gridspec.GridSpec(24, 2)
plt.figure(figsize=(24,24*8))

good = float_df['loan_status'] == 1
bad = float_df['loan_status'] == 0

for n, col in enumerate(loan_df[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(float_df[col][good], bins = 50, color='b')
    sns.distplot(float_df[col][bad], bins = 50, color='r')
    ax.set_ylabel('density',)
    ax.set_title(str(col), fontsize = 20)
    ax.set_xlabel('')
plt.show()

```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_69_0.png)



```python
plt.figure(figsize=(4,3))

plt.scatter(frame['annual_inc'], frame['funded_amnt'])
plt.title("Annual Income vs Funded Amount")
plt.ylabel('Funded Amount')
plt.xlabel('Annual Income')
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_70_0.png)



```python
plt.figure(figsize=(2,5))
sns.countplot(x='loan_status', data=loan_df, alpha = 0.8)
plt.title('Loan status', fontsize = 15)
plt.xlabel('')
plt.ylabel('Number of Loans')
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_71_0.png)



```python
plt.figure(figsize=(4,4))
sns.factorplot(x='home_ownership',hue='grade',
               data = loan_df,kind = 'count',size =6,
               hue_order = ['A','B','C','D','E','F','G'])
#plt.xlabel("Home Ownership")
plt.ylabel("Number of Loans")
plt.show()
```


    <matplotlib.figure.Figure at 0x285b73ba470>



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_72_1.png)



```python
#plt.figure(figsize=(6,6))

sns.set(style="white")

# Compute the correlation matrix
corr = float_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))


# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_73_0.png)


# Handling categorical features


```python
object_df = loan_df.select_dtypes(include=['object'])
object_df.shape
```




    (883689, 18)




```python
#Getting column names of object df with more than 10 unique values
# this is done to avoid a lot of dummy variable columns
for c in object_df.columns:
    if(object_df[c].nunique()>15):
        print(object_df[c].name)
        #print(object_df[c].value_counts())
        print()
```

    sub_grade

    issue_d

    zip_code

    addr_state

    earliest_cr_line

    revol_util

    last_pymnt_d

    last_credit_pull_d




```python
object_df.shape
```




    (883689, 18)




```python
#Dropping sub_grade and emp_title
#object_df = object_df.drop(['sub_grade', 'emp_title'], axis = 1)
#object_df.shape
```


```python
loan_df.drop(object_df, axis=1, inplace=True)
obj_df = pd.get_dummies(object_df)
loan_df = pd.concat([loan_df, obj_df], axis = 1)
```


```python
#loan_df.drop(['sub_grade', 'emp_title'], axis = 1, inplace = True)
```


```python
y = loan_df['loan_status']
X = loan_df.drop(['loan_status'], axis = 1)
print(y.shape)
print(X.shape)
```

    (883689,)
    (883689, 3478)



```python
X.dtypes.value_counts()
```




    float64    44
    uint8      43
    int32      3
    dtype: int64




```python
# Splitting 20% as test data (cross-validation set) and 80% as train data
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=0)
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (25439, 90)
    (25439,)
    (6360, 90)
    (6360,)



```python
print(pd.value_counts(pd.Series(y_train)))
print(' ')
print(pd.value_counts(pd.Series(y_test)))

#1 is fully paid
#0 is charged-off
```

    1    18929
    0    6510
    dtype: int64

    1    4770
    0    1590
    dtype: int64


# We scale the values first and then oversample. Scale train and test dataset separately. Make sure to scale the unseen test set before prediction.
# We need not scale the target variable in training and test sets


```python
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

print(X_train_scaled.shape)
print(' ')
print(X_test_scaled.shape)

```

    (25439, 90)

    (6360, 90)


# Oversampling using SMOTE technique

We perform oversampling of minority class observations to create a dataset with balanced classes. We do this only to the training dataset and not to the cross-validation set so that there is no data leakage. By doing this, we get a better representation of precision and recall scores in CV and the new unseen test set.


```python
X_train, y_train = SMOTE().fit_sample(X_train_scaled, y_train)
```

object_df = loan_df.select_dtypes(include=['object'])
cat_features = list(object_df.columns.values)
X = pd.get_dummies(X[X_clean.columns[:-2]], columns=cat_features).astype(float)## Scaling the values

# Checking shape of dataframes


```python
print(X_train_scaled.shape)
print(X_test_scaled.shape)

print(X_train.shape)

print(pd.value_counts(pd.Series(y_train)))
print(y_train.shape)
print(pd.value_counts(pd.Series(y_test)))

print(X_test.shape)
print(y_test.shape)

print(pd.value_counts(pd.Series(y_test)))
```

    (25439, 90)
    (6360, 90)
    (37858, 90)
    1    18929
    0    18929
    dtype: int64
    (37858,)
    1    4770
    0    1590
    dtype: int64
    (6360, 90)
    (6360,)
    1    4770
    0    1590
    dtype: int64


# Classification using LR


```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf_lr = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
```


```python
lr_model = clf_lr.fit(X_train, y_train)
```


```python
lr_model.best_params_
```




    {'C': 0.01}




```python
class_names = ['Good Loan', 'Bad Loan']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
def plotConfusion(model, X, y):
    y_true, y_pred = y, model.predict(X)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    print("Accuracy score in the dataset: ", accuracy_score(y_true,y_pred))
    print("Recall metric in the dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    print("ROC AUC score in the dataset: ", roc_auc_score(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
```


```python
plotConfusion(lr_model, X_train, y_train)
```

    Accuracy score in the dataset:  0.6689207036821808
    Recall metric in the dataset:  0.6362723862855936
    Confusion matrix, without normalization
    [[13280  5649]
     [ 6885 12044]]
    Normalized confusion matrix
    [[0.7  0.3 ]
     [0.36 0.64]]



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_100_1.png)



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_100_2.png)


    ROC AUC score in the dataset:  0.6689207036821808



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_100_4.png)



```python
plotConfusion(lr_model, X_test_scaled, y_test)
```

    Accuracy score in the dataset:  0.6496855345911949
    Recall metric in the dataset:  0.6410901467505241
    Confusion matrix, without normalization
    [[1074  516]
     [1712 3058]]
    Normalized confusion matrix
    [[0.68 0.32]
     [0.36 0.64]]



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_101_1.png)



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_101_2.png)


    ROC AUC score in the dataset:  0.6582809224318659



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_101_4.png)



```python
y_true, y_pred = y_test, lr_model.predict(X_test_scaled)
```


```python
pd.Series(y_true).value_counts()
```




    1    4770
    0    1590
    dtype: int64




```python
pd.Series(y_pred).value_counts()
```




    1    3574
    0    2786
    dtype: int64




```python
class1_prob = lr_model.predict_proba(X_test_scaled)

class1_prob = pd.DataFrame(class1_prob)
class1_prob = class1_prob.sort_values(by = [0])
class1_prob = class1_prob[0]
```


```python
class2_prob = np.where(class1_prob > 0.5, 1, 0)
plt.scatter(np.arange(X_test_scaled.shape[0]), class2_prob, s = 10, alpha = 0.1)
plt.plot(class1_prob.values, color = 'red')
plt.axhline(0.5, color='b')
plt.xlabel('number of observations')
plt.ylabel('Probability')
plt.title('Logistic function for Lending club')
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_106_0.png)



```python
#new_X_train = X_train[:500,:]
#new_y_train = y_train[:500]

clf2 = LogisticRegression(C= (lr_model.best_params_['C']))
clf2.fit(X_test_scaled, y_test)
#logreg = clf_lr.fit(X_test_scaled)
coeff = clf2.coef_
coeff_df = pd.DataFrame(coeff)
coeff_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.05195</td>
      <td>-0.05195</td>
      <td>-0.078421</td>
      <td>-0.153445</td>
      <td>-0.061975</td>
      <td>0.036764</td>
      <td>-0.004513</td>
      <td>-0.198198</td>
      <td>0.015722</td>
      <td>-0.01234</td>
      <td>...</td>
      <td>-0.035369</td>
      <td>-0.00554</td>
      <td>0.027047</td>
      <td>-0.010591</td>
      <td>-0.000701</td>
      <td>0.000701</td>
      <td>0.00111</td>
      <td>-0.00111</td>
      <td>-0.007899</td>
      <td>0.007899</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 90 columns</p>
</div>




```python
#feat_df = pd.DataFrame(new_X_train)
X = pd.DataFrame(X)
column_df = pd.DataFrame(X.columns)

```


```python
column_df.rename(columns = {0: 'feats'}, inplace = True)
coeff_dft = coeff_df.T
coeff_dft.rename(columns = {0:'feats'}, inplace = True)
feat_importance_df = pd.concat([column_df, coeff_dft], axis = 1)
feat_importance_df.columns = ['features', 'coefficients']
feat_importance_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>-0.051950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt</td>
      <td>-0.051950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>term</td>
      <td>-0.078421</td>
    </tr>
    <tr>
      <th>3</th>
      <td>int_rate</td>
      <td>-0.153445</td>
    </tr>
    <tr>
      <th>4</th>
      <td>installment</td>
      <td>-0.061975</td>
    </tr>
    <tr>
      <th>5</th>
      <td>emp_length</td>
      <td>0.036764</td>
    </tr>
    <tr>
      <th>6</th>
      <td>annual_inc</td>
      <td>-0.004513</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dti</td>
      <td>-0.198198</td>
    </tr>
    <tr>
      <th>8</th>
      <td>inq_last_6mths</td>
      <td>0.015722</td>
    </tr>
    <tr>
      <th>9</th>
      <td>open_acc</td>
      <td>-0.012340</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pub_rec</td>
      <td>-0.026793</td>
    </tr>
    <tr>
      <th>11</th>
      <td>revol_bal</td>
      <td>0.030406</td>
    </tr>
    <tr>
      <th>12</th>
      <td>revol_util</td>
      <td>-0.065573</td>
    </tr>
    <tr>
      <th>13</th>
      <td>total_acc</td>
      <td>0.059570</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tot_cur_bal</td>
      <td>0.053442</td>
    </tr>
    <tr>
      <th>15</th>
      <td>open_il_24m</td>
      <td>0.016101</td>
    </tr>
    <tr>
      <th>16</th>
      <td>total_bal_il</td>
      <td>-0.062862</td>
    </tr>
    <tr>
      <th>17</th>
      <td>max_bal_bc</td>
      <td>0.026819</td>
    </tr>
    <tr>
      <th>18</th>
      <td>all_util</td>
      <td>-0.020104</td>
    </tr>
    <tr>
      <th>19</th>
      <td>total_rev_hi_lim</td>
      <td>0.027979</td>
    </tr>
    <tr>
      <th>20</th>
      <td>avg_cur_bal</td>
      <td>0.043406</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bc_open_to_buy</td>
      <td>0.024996</td>
    </tr>
    <tr>
      <th>22</th>
      <td>bc_util</td>
      <td>0.066377</td>
    </tr>
    <tr>
      <th>23</th>
      <td>chargeoff_within_12_mths</td>
      <td>-0.005195</td>
    </tr>
    <tr>
      <th>24</th>
      <td>mo_sin_old_il_acct</td>
      <td>0.012971</td>
    </tr>
    <tr>
      <th>25</th>
      <td>mo_sin_old_rev_tl_op</td>
      <td>0.017635</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mo_sin_rcnt_rev_tl_op</td>
      <td>0.053837</td>
    </tr>
    <tr>
      <th>27</th>
      <td>mo_sin_rcnt_tl</td>
      <td>-0.046803</td>
    </tr>
    <tr>
      <th>28</th>
      <td>mort_acc</td>
      <td>0.077158</td>
    </tr>
    <tr>
      <th>29</th>
      <td>mths_since_recent_bc</td>
      <td>-0.000602</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>purpose_car</td>
      <td>0.019965</td>
    </tr>
    <tr>
      <th>61</th>
      <td>purpose_credit_card</td>
      <td>0.008024</td>
    </tr>
    <tr>
      <th>62</th>
      <td>purpose_debt_consolidation</td>
      <td>0.002420</td>
    </tr>
    <tr>
      <th>63</th>
      <td>purpose_home_improvement</td>
      <td>0.014230</td>
    </tr>
    <tr>
      <th>64</th>
      <td>purpose_house</td>
      <td>0.022374</td>
    </tr>
    <tr>
      <th>65</th>
      <td>purpose_major_purchase</td>
      <td>0.003545</td>
    </tr>
    <tr>
      <th>66</th>
      <td>purpose_medical</td>
      <td>-0.035369</td>
    </tr>
    <tr>
      <th>67</th>
      <td>purpose_moving</td>
      <td>-0.005540</td>
    </tr>
    <tr>
      <th>68</th>
      <td>purpose_other</td>
      <td>-0.017834</td>
    </tr>
    <tr>
      <th>69</th>
      <td>purpose_renewable_energy</td>
      <td>-0.000572</td>
    </tr>
    <tr>
      <th>70</th>
      <td>purpose_small_business</td>
      <td>-0.027292</td>
    </tr>
    <tr>
      <th>71</th>
      <td>purpose_vacation</td>
      <td>-0.014169</td>
    </tr>
    <tr>
      <th>72</th>
      <td>title_Business</td>
      <td>-0.027292</td>
    </tr>
    <tr>
      <th>73</th>
      <td>title_Car financing</td>
      <td>0.019965</td>
    </tr>
    <tr>
      <th>74</th>
      <td>title_Credit card refinancing</td>
      <td>-0.004439</td>
    </tr>
    <tr>
      <th>75</th>
      <td>title_Debt consolidation</td>
      <td>0.002132</td>
    </tr>
    <tr>
      <th>76</th>
      <td>title_Green loan</td>
      <td>-0.000572</td>
    </tr>
    <tr>
      <th>77</th>
      <td>title_Home buying</td>
      <td>0.000847</td>
    </tr>
    <tr>
      <th>78</th>
      <td>title_Home improvement</td>
      <td>-0.004230</td>
    </tr>
    <tr>
      <th>79</th>
      <td>title_Major purchase</td>
      <td>0.003545</td>
    </tr>
    <tr>
      <th>80</th>
      <td>title_Medical expenses</td>
      <td>-0.035369</td>
    </tr>
    <tr>
      <th>81</th>
      <td>title_Moving and relocation</td>
      <td>-0.005540</td>
    </tr>
    <tr>
      <th>82</th>
      <td>title_Other</td>
      <td>0.027047</td>
    </tr>
    <tr>
      <th>83</th>
      <td>title_Vacation</td>
      <td>-0.010591</td>
    </tr>
    <tr>
      <th>84</th>
      <td>initial_list_status_f</td>
      <td>-0.000701</td>
    </tr>
    <tr>
      <th>85</th>
      <td>initial_list_status_w</td>
      <td>0.000701</td>
    </tr>
    <tr>
      <th>86</th>
      <td>application_type_Individual</td>
      <td>0.001110</td>
    </tr>
    <tr>
      <th>87</th>
      <td>application_type_Joint App</td>
      <td>-0.001110</td>
    </tr>
    <tr>
      <th>88</th>
      <td>disbursement_method_Cash</td>
      <td>-0.007899</td>
    </tr>
    <tr>
      <th>89</th>
      <td>disbursement_method_DirectPay</td>
      <td>0.007899</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 2 columns</p>
</div>




```python
feat_df = feat_importance_df.sort_values(by=['coefficients'], ascending = False)
top20_feat = feat_df[:20]


f, ax = plt.subplots(figsize=(10, 12))

ax = sns.barplot(y='features', x='coefficients', data = top20_feat)
ax.set_ylabel("features", fontsize=15)
ax.set_xlabel("Coeffcients = increase in logodds for 1 unit increase in value", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()
```


![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_110_0.png)



```python
clf2.coef_
```




    array([[-0.05, -0.05, -0.11,  0.2 , -0.11,  0.05, -0.07, -0.28,  0.03,
            -0.52, -0.05, -0.13, -0.08,  0.46, -0.23,  0.02, -0.73,  0.06,
            -0.  , -0.07,  0.03,  0.17,  0.14, -0.01,  0.01,  0.02,  0.06,
            -0.08,  0.01,  0.01,  0.21,  0.09, -0.22, -0.06, -0.27,  0.29,
            -0.15, -0.47,  0.43, -0.14,  0.01, -0.13,  0.44,  0.58, -0.04,
             0.27, -0.61,  0.1 ,  0.01, -0.07, -0.02, -0.05,  0.01,  0.04,
             0.09, -0.03, -0.07, -0.01,  0.01, -0.01,  0.02,  0.09,  0.14,
             0.13,  0.2 ,  0.  , -0.05, -0.01, -0.55, -0.  , -0.03, -0.07,
            -0.03,  0.02, -0.07, -0.12, -0.  , -0.19, -0.13,  0.  , -0.05,
            -0.01,  0.55,  0.06,  0.01, -0.01, -0.01,  0.01, -0.01,  0.01,
             0.69, -0.69]])




```python
clf2.intercept_
```




    array([1.21])




```python
logodds = clf2.intercept_ + clf2.coef_
odds = np.exp(logodds)
prob = odds/(1+odds)
prob
```




    array([[0.76, 0.76, 0.75, 0.8 , 0.75, 0.78, 0.76, 0.72, 0.77, 0.67, 0.76,
            0.75, 0.75, 0.84, 0.73, 0.77, 0.62, 0.78, 0.77, 0.76, 0.78, 0.8 ,
            0.79, 0.77, 0.77, 0.77, 0.78, 0.76, 0.77, 0.77, 0.81, 0.79, 0.73,
            0.76, 0.72, 0.82, 0.74, 0.68, 0.84, 0.74, 0.77, 0.75, 0.84, 0.86,
            0.76, 0.81, 0.65, 0.79, 0.77, 0.76, 0.77, 0.76, 0.77, 0.78, 0.79,
            0.76, 0.76, 0.77, 0.77, 0.77, 0.77, 0.78, 0.79, 0.79, 0.8 , 0.77,
            0.76, 0.77, 0.66, 0.77, 0.76, 0.76, 0.76, 0.77, 0.76, 0.75, 0.77,
            0.74, 0.75, 0.77, 0.76, 0.77, 0.85, 0.78, 0.77, 0.77, 0.77, 0.77,
            0.77, 0.77, 0.87, 0.63]])



# Linear discriminant Analysis


```python
lda = LinearDiscriminantAnalysis()
```


```python
clf_lda = lda.fit(X_train,y_train)
```

    C:\Anaconda\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")



```python
class_names = ['Good Loan', 'Bad Loan']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
```


```python
def plotConfusion(model, X, y):
    y_true, y_pred = y, model.predict(X)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    print("Accuracy score in the dataset: ", accuracy_score(y_true,y_pred))
    print("Recall metric in the dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    print("ROC AUC score in the dataset: ", roc_auc_score(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
```


```python
plotConfusion(clf_lda, X_train, y_train)
```

    Accuracy score in the dataset:  0.6677848803423319
    Recall metric in the dataset:  0.6357440963600824
    Confusion matrix, without normalization
    [[13247  5682]
     [ 6895 12034]]
    Normalized confusion matrix
    [[0.7  0.3 ]
     [0.36 0.64]]



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_119_1.png)



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_119_2.png)


    ROC AUC score in the dataset:  0.6677848803423319



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_119_4.png)



```python
plotConfusion(clf_lda, X_test_scaled, y_test)
```

    Accuracy score in the dataset:  0.6787735849056604
    Recall metric in the dataset:  0.6853249475890986
    Confusion matrix, without normalization
    [[1048  542]
     [1501 3269]]
    Normalized confusion matrix
    [[0.66 0.34]
     [0.31 0.69]]



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_120_1.png)



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_120_2.png)


    ROC AUC score in the dataset:  0.6722222222222223



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_120_4.png)



```python
new_X_train = X_train[:50,:]
new_y_train = y_train[:50]
```


```python
new_X_train[:,0]
```




    array([ 0.15, -0.29, -1.31, -0.07,  1.16, -0.03,  2.27, -0.52, -0.76,
            0.38,  2.27,  1.05, -0.29,  1.98,  0.79, -0.29, -0.22, -0.52,
            2.27,  2.27,  2.83,  0.38,  0.15,  0.04,  0.6 ,  0.2 , -0.29,
            0.04, -0.29, -1.08, -0.52,  0.04,  0.04,  1.71,  2.83, -1.32,
            2.27,  2.27,  0.6 , -1.41, -0.74, -0.52, -0.29,  1.71,  0.38,
           -1.08,  1.58,  0.6 ,  1.49, -1.1 ])




```python
new_X_train.shape
```




    (50, 91)




```python
new_y_train
```




    array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 0, 1, 1, 1, 1], dtype=int64)




```python
x1 = pd.DataFrame(new_X_train)
y1 = new_y_train
```


```python
lda3 = LinearDiscriminantAnalysis()
clf_lda3 = lda3.fit(x1, y1)
```

    C:\Anaconda\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")



```python
label_dict = {1: 'Good', 0: 'Bad'}
```


```python
def plot_scikit_lda(arg):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        np.arange(2),('o', 'o'),('red', 'green')):

        plt.scatter(x=arg[:,0][new_y_train == label],
                    y=arg[:,0][new_y_train == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()
```


```python
clf_lda2.mean()
```




    -0.07754277405242445




```python
x1 = pd.DataFrame(new_X_train)
y1 = new_y_train
```


```python
type(y1)
```




    numpy.ndarray




```python
lda3 = LinearDiscriminantAnalysis()
clf_lda3 = lda3.fit(x1, y1)
```

    C:\Anaconda\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")


# Plotting the fitted LDA model with respect to two columns


```python
def plotModel(model, x, y, label):
    '''
    model: a fitted model
    x, y: two variables, should arrays
    label: true label
    '''
    x_min = x.min() - 1
    x_max = x.max() + 1
    y_min = y.min() - 1
    y_max = y.max() + 1
    import  matplotlib.pyplot as plt
    from matplotlib import colors
    colDict = {'red': [(0, 1, 1), (1, 0.7, 0.7)],
               'green': [(0, 1, 0.5), (1, 0.7, 0.7)],
               'blue': [(0, 1, 0.5), (1, 1, 1)]}
    cmap = colors.LinearSegmentedColormap('red_blue_classes', colDict)
    plt.cm.register_cmap(cmap=cmap)
    nx, ny = 200, 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ## plot colormap
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes')
    ## plot boundaries
    plt.contour(xx, yy, Z, [0.5], linewidths=1., colors='k')
    plt.contour(xx, yy, Z, [1], linewidths=1., colors='k')
    ## plot scatters ans true labels
    plt.scatter(x, y, c = label)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

labels = x1.columns.values

def pairPlot(model, i, j):
    model.fit(x1.iloc[:,[i,j]], y1)
    plotModel(model, x1.iloc[:, i], x1.iloc[:, j], y1)
    plt.xlabel(labels[i])
    plt.ylabel(labels[j])
```


```python
plt.rcParams['figure.figsize'] = 16, 10
plt.subplot(321)
pairPlot(clf_lda3,28,33)

plt.subplot(322)
pairPlot(clf_lda3,2,5)

plt.subplot(323)
pairPlot(clf_lda3,2,7)

plt.subplot(324)
pairPlot(clf_lda3,6,30)

plt.subplot(325)
pairPlot(clf_lda3,23,36)

plt.subplot(326)
pairPlot(clf_lda3,40,58)

```

    C:\Anaconda\Anaconda3\lib\site-packages\matplotlib\contour.py:1180: UserWarning: No contour levels were found within the data range.
      warnings.warn("No contour levels were found"



![png](lendingClub_LRandLDA_files/lendingClub_LRandLDA_135_1.png)
