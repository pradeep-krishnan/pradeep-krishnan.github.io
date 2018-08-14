---
title: "Datascience project4"
excerpt: "using python, scikitlearn, this project tries to demonstrate my expertise in working with machine learning packages"
modified: 2016-06-01T16:28:11-04:00
tags: [python, pandas, pca]
comments: true
share: true
read_time: true
header:
  teaser: /images/img1.jpg
---
## How do we measure the similarity of names in terms of the evolution of their use over time?

Analysis of the similarity of names in terms of its use.

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
