---
title: "Ginger Gulp Identity"
excerpt: "Ginger Gulp design"
date: 2018-08-14

---
## How do we measure the similarity of names in terms of the evolution of their use over time?
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
