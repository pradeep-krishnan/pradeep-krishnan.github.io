---
layout: archive
title: "House Price prediction"
excerpt: "Regression, Ensembling, Stacking"
modified: 2016-08-15
tags: [python, pandas, scikit-learn, regression]
comments: true
read_time: true
header:
  teaser: /images/img6.jpg
---
## Predicting house prices using regression modeling
Topics covered in this post:
Multiple Linear Regression
Lasso
Ridge
Decision Tree
Random Forest

```python
import os
import numpy as np
import pandas
pandas.set_option('max_rows', 10)

folder = os.getcwd()
file_name = os.path.join(folder, 'historico-nombres.csv')

df = pandas.read_csv(file_name)
df.columns = ['name', 'amount', 'year']
df.head()
```
You can see the image below:
![alt]({{ site.url }}{{ site.baseurl }}/images/img3.jpg)
Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.

```python
import os
import numpy as np
```
project starts here ;)
