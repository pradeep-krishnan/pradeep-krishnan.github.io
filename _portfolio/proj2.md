---
layout: article
title: Name temporal similarity
categories: projects
modified: 2016-06-01T16:28:11-04:00
tags: [python, pandas, pca]
comments: true
share: true
read_time: true
image:
  header: image/img2.jpg
---

Analysis of the similarity of names in terms of its use evolution in Argentina between 1922 and 2015.

## How do we measure the similarity of names in terms of the evolution of their use over time?

This is the question that kickstarted this small toy project. It gave me a nice excuse to take my first steps into the pandas project. We took data from Argentina's public data portal so the results are only applicable there. The ideas could be adapted to any population if the data is available.

### Data loading and preprocessing

Naming by year in Argentina dataset taken from [this website](http://www.datos.gob.ar/dataset/nombres-personas-fisicas). Here we assume the file `historico-nombres.csv` is placed in the same folder as the python script.


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
