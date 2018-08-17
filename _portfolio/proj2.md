---
title: "Loan default prediction"
excerpt: "Classification techniques"
modified: 2016-08-15T16:28:11-04:00
tags: [python, pandas, scikit-learn]
comments: true
share: false
read_time: true
header:
  teaser: /images/img3.jpg
---

Predicting probability of default of lending club loans

### Data loading and preprocessing



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
