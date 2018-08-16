---
title: "Datascience project5"
date: 2018-08-14
header:
  teaser: /images/img4.jpg
---
## How do we measure the similarity of names in terms of the evolution of their use over time?

Analysis of the similarity of names in terms of its use.

```python

# Allstate Claims Severity


```python
import warnings
warnings.filterwarnings('ignore')

import zipfile
import pandas as pd

import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None
```


```python
#extract all zipped files
with zipfile.ZipFile("train.csv.zip","r") as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("test.csv.zip","r") as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("sample_submission.csv.zip","r") as zip_ref:
    zip_ref.extractall()
```


```python
#reading all the files as dataframes
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")
```

![alt]({{ site.url }}{{ site.baseurl }}/images/sample11.png)




```python
print("shape of train: ", train.shape)
print("shape of test: ",test.shape)
print("shape of submission: ",submission.shape)
```

    shape of train:  (188318, 132)
    shape of test:  (125546, 131)
    shape of submission:  (125546, 2)



```python
#getting summary statistics of columns
train.describe()
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
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
      <td>188318.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>294135.982561</td>
      <td>0.493861</td>
      <td>0.507188</td>
      <td>0.498918</td>
      <td>0.491812</td>
      <td>0.487428</td>
      <td>0.490945</td>
      <td>0.484970</td>
      <td>0.486437</td>
      <td>0.485506</td>
      <td>0.498066</td>
      <td>0.493511</td>
      <td>0.493150</td>
      <td>0.493138</td>
      <td>0.495717</td>
      <td>3037.337686</td>
    </tr>
    <tr>
      <th>std</th>
      <td>169336.084867</td>
      <td>0.187640</td>
      <td>0.207202</td>
      <td>0.202105</td>
      <td>0.211292</td>
      <td>0.209027</td>
      <td>0.205273</td>
      <td>0.178450</td>
      <td>0.199370</td>
      <td>0.181660</td>
      <td>0.185877</td>
      <td>0.209737</td>
      <td>0.209427</td>
      <td>0.212777</td>
      <td>0.222488</td>
      <td>2904.086186</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000016</td>
      <td>0.001149</td>
      <td>0.002634</td>
      <td>0.176921</td>
      <td>0.281143</td>
      <td>0.012683</td>
      <td>0.069503</td>
      <td>0.236880</td>
      <td>0.000080</td>
      <td>0.000000</td>
      <td>0.035321</td>
      <td>0.036232</td>
      <td>0.000228</td>
      <td>0.179722</td>
      <td>0.670000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>147748.250000</td>
      <td>0.346090</td>
      <td>0.358319</td>
      <td>0.336963</td>
      <td>0.327354</td>
      <td>0.281143</td>
      <td>0.336105</td>
      <td>0.350175</td>
      <td>0.312800</td>
      <td>0.358970</td>
      <td>0.364580</td>
      <td>0.310961</td>
      <td>0.311661</td>
      <td>0.315758</td>
      <td>0.294610</td>
      <td>1204.460000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>294539.500000</td>
      <td>0.475784</td>
      <td>0.555782</td>
      <td>0.527991</td>
      <td>0.452887</td>
      <td>0.422268</td>
      <td>0.440945</td>
      <td>0.438285</td>
      <td>0.441060</td>
      <td>0.441450</td>
      <td>0.461190</td>
      <td>0.457203</td>
      <td>0.462286</td>
      <td>0.363547</td>
      <td>0.407403</td>
      <td>2115.570000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>440680.500000</td>
      <td>0.623912</td>
      <td>0.681761</td>
      <td>0.634224</td>
      <td>0.652072</td>
      <td>0.643315</td>
      <td>0.655021</td>
      <td>0.591045</td>
      <td>0.623580</td>
      <td>0.566820</td>
      <td>0.614590</td>
      <td>0.678924</td>
      <td>0.675759</td>
      <td>0.689974</td>
      <td>0.724623</td>
      <td>3864.045000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>587633.000000</td>
      <td>0.984975</td>
      <td>0.862654</td>
      <td>0.944251</td>
      <td>0.954297</td>
      <td>0.983674</td>
      <td>0.997162</td>
      <td>1.000000</td>
      <td>0.980200</td>
      <td>0.995400</td>
      <td>0.994980</td>
      <td>0.998742</td>
      <td>0.998484</td>
      <td>0.988494</td>
      <td>0.844848</td>
      <td>121012.250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#datatypes of columns
print("Datatypes of columns: ")
print(train.dtypes.value_counts())
print('\n')
print("Datatype of response variable: ", train['loss'].dtype)
print("Datatype of id: ", train['id'].dtype)
```

    Datatypes of columns:
    object     116
    float64     15
    int64        1
    dtype: int64


    Datatype of response variable:  float64
    Datatype of id:  int64



```python
# print all columns in dataframe
pd.set_option('display.max_columns', None)

train.head()
```

![alt]({{ site.url }}{{ site.baseurl }}/images/sample12.png)



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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>cat11</th>
      <th>cat12</th>
      <th>cat13</th>
      <th>cat14</th>
      <th>cat15</th>
      <th>cat16</th>
      <th>cat17</th>
      <th>cat18</th>
      <th>cat19</th>
      <th>cat20</th>
      <th>cat21</th>
      <th>cat22</th>
      <th>cat23</th>
      <th>cat24</th>
      <th>cat25</th>
      <th>cat26</th>
      <th>cat27</th>
      <th>cat28</th>
      <th>cat29</th>
      <th>cat30</th>
      <th>cat31</th>
      <th>cat32</th>
      <th>cat33</th>
      <th>cat34</th>
      <th>cat35</th>
      <th>cat36</th>
      <th>cat37</th>
      <th>cat38</th>
      <th>cat39</th>
      <th>cat40</th>
      <th>cat41</th>
      <th>cat42</th>
      <th>cat43</th>
      <th>cat44</th>
      <th>cat45</th>
      <th>cat46</th>
      <th>cat47</th>
      <th>cat48</th>
      <th>cat49</th>
      <th>cat50</th>
      <th>cat51</th>
      <th>cat52</th>
      <th>cat53</th>
      <th>cat54</th>
      <th>cat55</th>
      <th>cat56</th>
      <th>cat57</th>
      <th>cat58</th>
      <th>cat59</th>
      <th>cat60</th>
      <th>cat61</th>
      <th>cat62</th>
      <th>cat63</th>
      <th>cat64</th>
      <th>cat65</th>
      <th>cat66</th>
      <th>cat67</th>
      <th>cat68</th>
      <th>cat69</th>
      <th>cat70</th>
      <th>cat71</th>
      <th>cat72</th>
      <th>cat73</th>
      <th>cat74</th>
      <th>cat75</th>
      <th>cat76</th>
      <th>cat77</th>
      <th>cat78</th>
      <th>cat79</th>
      <th>cat80</th>
      <th>cat81</th>
      <th>cat82</th>
      <th>cat83</th>
      <th>cat84</th>
      <th>cat85</th>
      <th>cat86</th>
      <th>cat87</th>
      <th>cat88</th>
      <th>cat89</th>
      <th>cat90</th>
      <th>cat91</th>
      <th>cat92</th>
      <th>cat93</th>
      <th>cat94</th>
      <th>cat95</th>
      <th>cat96</th>
      <th>cat97</th>
      <th>cat98</th>
      <th>cat99</th>
      <th>cat100</th>
      <th>cat101</th>
      <th>cat102</th>
      <th>cat103</th>
      <th>cat104</th>
      <th>cat105</th>
      <th>cat106</th>
      <th>cat107</th>
      <th>cat108</th>
      <th>cat109</th>
      <th>cat110</th>
      <th>cat111</th>
      <th>cat112</th>
      <th>cat113</th>
      <th>cat114</th>
      <th>cat115</th>
      <th>cat116</th>
      <th>cont1</th>
      <th>cont2</th>
      <th>cont3</th>
      <th>cont4</th>
      <th>cont5</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>C</td>
      <td>E</td>
      <td>A</td>
      <td>C</td>
      <td>T</td>
      <td>B</td>
      <td>G</td>
      <td>A</td>
      <td>A</td>
      <td>I</td>
      <td>E</td>
      <td>G</td>
      <td>J</td>
      <td>G</td>
      <td>BU</td>
      <td>BC</td>
      <td>C</td>
      <td>AS</td>
      <td>S</td>
      <td>A</td>
      <td>O</td>
      <td>LB</td>
      <td>0.726300</td>
      <td>0.245921</td>
      <td>0.187583</td>
      <td>0.789639</td>
      <td>0.310061</td>
      <td>0.718367</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>D</td>
      <td>C</td>
      <td>E</td>
      <td>E</td>
      <td>D</td>
      <td>T</td>
      <td>L</td>
      <td>F</td>
      <td>A</td>
      <td>A</td>
      <td>E</td>
      <td>E</td>
      <td>I</td>
      <td>K</td>
      <td>K</td>
      <td>BI</td>
      <td>CQ</td>
      <td>A</td>
      <td>AV</td>
      <td>BM</td>
      <td>A</td>
      <td>O</td>
      <td>DP</td>
      <td>0.330514</td>
      <td>0.737068</td>
      <td>0.592681</td>
      <td>0.614134</td>
      <td>0.885834</td>
      <td>0.438917</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>C</td>
      <td>B</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>D</td>
      <td>C</td>
      <td>E</td>
      <td>E</td>
      <td>A</td>
      <td>D</td>
      <td>L</td>
      <td>O</td>
      <td>A</td>
      <td>B</td>
      <td>E</td>
      <td>F</td>
      <td>H</td>
      <td>F</td>
      <td>A</td>
      <td>AB</td>
      <td>DK</td>
      <td>A</td>
      <td>C</td>
      <td>AF</td>
      <td>A</td>
      <td>I</td>
      <td>GK</td>
      <td>0.261841</td>
      <td>0.358319</td>
      <td>0.484196</td>
      <td>0.236924</td>
      <td>0.397069</td>
      <td>0.289648</td>
      <td>0.315545</td>
      <td>0.27320</td>
      <td>0.26076</td>
      <td>0.32446</td>
      <td>0.381398</td>
      <td>0.373424</td>
      <td>0.195709</td>
      <td>0.774425</td>
      <td>3005.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>B</td>
      <td>D</td>
      <td>D</td>
      <td>D</td>
      <td>B</td>
      <td>C</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>D</td>
      <td>C</td>
      <td>E</td>
      <td>E</td>
      <td>D</td>
      <td>T</td>
      <td>I</td>
      <td>D</td>
      <td>A</td>
      <td>A</td>
      <td>E</td>
      <td>E</td>
      <td>I</td>
      <td>K</td>
      <td>K</td>
      <td>BI</td>
      <td>CS</td>
      <td>C</td>
      <td>N</td>
      <td>AE</td>
      <td>A</td>
      <td>O</td>
      <td>DJ</td>
      <td>0.321594</td>
      <td>0.555782</td>
      <td>0.527991</td>
      <td>0.373816</td>
      <td>0.422268</td>
      <td>0.440945</td>
      <td>0.391128</td>
      <td>0.31796</td>
      <td>0.32128</td>
      <td>0.44467</td>
      <td>0.327915</td>
      <td>0.321570</td>
      <td>0.605077</td>
      <td>0.602642</td>
      <td>939.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>B</td>
      <td>B</td>
      <td>C</td>
      <td>B</td>
      <td>B</td>
      <td>C</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>H</td>
      <td>D</td>
      <td>B</td>
      <td>D</td>
      <td>E</td>
      <td>E</td>
      <td>A</td>
      <td>P</td>
      <td>F</td>
      <td>J</td>
      <td>A</td>
      <td>A</td>
      <td>D</td>
      <td>E</td>
      <td>K</td>
      <td>G</td>
      <td>B</td>
      <td>H</td>
      <td>C</td>
      <td>C</td>
      <td>Y</td>
      <td>BM</td>
      <td>A</td>
      <td>K</td>
      <td>CK</td>
      <td>0.273204</td>
      <td>0.159990</td>
      <td>0.527991</td>
      <td>0.473202</td>
      <td>0.704268</td>
      <td>0.178193</td>
      <td>0.247408</td>
      <td>0.24564</td>
      <td>0.22089</td>
      <td>0.21230</td>
      <td>0.204687</td>
      <td>0.202213</td>
      <td>0.246011</td>
      <td>0.432606</td>
      <td>2763.85</td>
    </tr>
  </tbody>
</table>
</div>




```python
#From above, we note that we can remove id column
#we have to check the number of unique values in categorical column
```


```python
train.drop('id', axis = 1, inplace = True)
```


```python
pd.reset_option("^display")
#check for missing values
sum(train.isnull().sum())
```




    0




```python
#no missing values
train.head()
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
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>cat10</th>
      <th>...</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>0.718367</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>0.438917</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>0.289648</td>
      <td>0.315545</td>
      <td>0.27320</td>
      <td>0.26076</td>
      <td>0.32446</td>
      <td>0.381398</td>
      <td>0.373424</td>
      <td>0.195709</td>
      <td>0.774425</td>
      <td>3005.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>...</td>
      <td>0.440945</td>
      <td>0.391128</td>
      <td>0.31796</td>
      <td>0.32128</td>
      <td>0.44467</td>
      <td>0.327915</td>
      <td>0.321570</td>
      <td>0.605077</td>
      <td>0.602642</td>
      <td>939.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>B</td>
      <td>...</td>
      <td>0.178193</td>
      <td>0.247408</td>
      <td>0.24564</td>
      <td>0.22089</td>
      <td>0.21230</td>
      <td>0.204687</td>
      <td>0.202213</td>
      <td>0.246011</td>
      <td>0.432606</td>
      <td>2763.85</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 131 columns</p>
</div>



# EDA

## Plotting response variable


```python
sns.distplot(train['loss'])
plt.show()
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_14_0.png)



```python
#Taking a closer look
fig = train['loss'].hist(bins = 100)
fig.set_xlim(0,20000)
```




    (0, 20000)




![png](allstate_severity_pradeep_files/allstate_severity_pradeep_15_1.png)



```python
#Log transformation of loss column

log_loss = np.log1p(train['loss'])

fig, (ax1, ax2) = plt.subplots(ncols = 2, sharey = True)
sns.violinplot(y=log_loss, ax = ax1)
sns.boxplot(y=log_loss, ax = ax2)
plt.show()
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_16_0.png)



```python
sns.distplot(np.log1p(train['loss']))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e031557b38>




![png](allstate_severity_pradeep_files/allstate_severity_pradeep_17_1.png)


## Plotting continuous features


```python
train_float_col = train.select_dtypes(include = ['float64'])
```


```python
train_float_col.drop('loss', axis = 1, inplace = True)
```


```python
#histogram of continuous features
fig = train_float_col.hist(layout = (4,4), figsize = (16,10), xrot = 90, bins = 20)
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_21_0.png)



```python
train_float_col.skew().sort_values(ascending = False)
#values close to zero show less skew
```




    cont9     1.072429
    cont7     0.826053
    cont5     0.681622
    cont8     0.676634
    cont1     0.516424
    cont6     0.461214
    cont4     0.416096
    cont13    0.380742
    cont10    0.355001
    cont12    0.291992
    cont11    0.280821
    cont14    0.248674
    cont3    -0.010002
    cont2    -0.310941
    dtype: float64




```python
#Boxplot of continuous features
plt.figure(figsize = (20,10))
fig = sns.boxplot(data = train_float_col)
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_23_0.png)



```python
#violin-plot of continuous features
plt.figure(figsize = (20,10))
fig = sns.violinplot(data = train_float_col)
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_24_0.png)



```python
#Finding correlation
sns.set(style="white")
train_float_col_corr = train_float_col.corr()

mask = np.zeros_like(train_float_col_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (8,8))
cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(train_float_col_corr, mask=mask, cmap=cmap, vmax=1, vmin = -1,
            center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_25_0.png)


#### As seen above, there are definitely highly correlated continuous features. Moving on to plotting them


```python
for i in range(0,14):
    for j in range(i+1,14):
        if(abs(train_float_col_corr.iloc[i,j]) >= 0.5):
            sns.regplot(train[train_float_col.columns[i]], train[train_float_col.columns[j]])
            plt.show()
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_0.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_1.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_2.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_3.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_4.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_5.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_6.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_7.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_8.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_9.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_10.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_11.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_12.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_13.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_14.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_15.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_16.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_17.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_18.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_19.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_20.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_21.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_27_22.png)


## Exploring Categorical features


```python
train_obj_col = train.select_dtypes(include = ['object'])
```


```python
for i in range(29):
    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8,5))
    for j in range(4):
        sns.countplot(train_obj_col.columns[i*4+j], data = train_obj_col, ax = ax[j])
```


![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_0.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_1.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_2.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_3.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_4.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_5.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_6.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_7.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_8.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_9.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_10.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_11.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_12.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_13.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_14.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_15.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_16.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_17.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_18.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_19.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_20.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_21.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_22.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_23.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_24.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_25.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_26.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_27.png)



![png](allstate_severity_pradeep_files/allstate_severity_pradeep_30_28.png)



```python
for col in train_obj_col.columns:
    if(train_obj_col[col].nunique()>20):
        print(train_obj_col[col].name,":", train_obj_col[col].nunique())
```

    cat109 : 84
    cat110 : 131
    cat112 : 51
    cat113 : 61
    cat115 : 23
    cat116 : 326


#### Encoding categorical data


```python
y = train['loss']
test.drop('id', axis = 1, inplace = True)
```


```python
one_hot_encoded_train = pd.get_dummies(train)
one_hot_encoded_test = pd.get_dummies(test)
final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,
                                                      join = 'inner', axis = 1)
```


```python
#final_test.drop('loss', axis = 1, inplace = True)
```


```python
final_test.dtypes.value_counts()
```




    uint8      1065
    float64      14
    dtype: int64




```python
#for col in train.columns:
    if(train[col].dtype == 'object'):
        print(train[col].name)
```


      File "<ipython-input-722-a475d945c0ba>", line 2
        if(train[col].dtype == 'object'):
        ^
    IndentationError: unexpected indent




```python
null_col_df = final_test.isnull().sum(axis=0).sort_values(ascending=False).reset_index()
null_col_df.columns = ['column_name', 'missing_count']
null_col_df
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
      <th>column_name</th>
      <th>missing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cat116_Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat104_D</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat105_A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat104_Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat104_P</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat104_O</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat104_N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat104_M</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat104_L</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cat104_K</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cat104_J</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cat104_I</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cat104_H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cat104_G</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cat104_F</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cat104_E</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cat104_C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>cat105_C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>cat104_B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>cat104_A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>cat103_N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>cat103_L</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>cat103_K</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>cat103_J</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>cat103_I</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>cat103_H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cat103_G</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>cat103_F</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cat103_E</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>cat103_D</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1049</th>
      <td>cat113_K</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1050</th>
      <td>cat113_J</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1051</th>
      <td>cat113_I</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1052</th>
      <td>cat113_H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1053</th>
      <td>cat113_G</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>cat113_F</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>cat113_E</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>cat113_C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>cat113_BO</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>cat113_BN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>cat113_BM</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>cat113_BL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>cat113_BK</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>cat113_BJ</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>cat113_BI</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>cat113_BH</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1065</th>
      <td>cat113_BG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>cat113_BF</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>cat113_BD</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>cat113_BC</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>cat113_BB</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>cat113_BA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>cat113_B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>cat113_AY</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>cat113_AX</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>cat113_AW</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1075</th>
      <td>cat113_AV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>cat113_AU</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1077</th>
      <td>cat113_AT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>cont1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1079 rows × 2 columns</p>
</div>




```python
final_train = pd.concat([final_train,y], axis = 1)
```
