---
title: "Beer recommendation system"
excerpt: "Recommender systems, NLP"
modified: 2016-08-18T16:28:11-04:00
tags: [python, pandas, nltk, NLP]
comments: true
share: false
read_time: true
header:
  teaser: /images/img3.jpg
---
#Beer recommendation
##Based on scraped dataset
###Flask app front end

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
Analysis of the similarity of names in terms of its use. Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.Analysis of the similarity of names in terms of its use.

```python
import os
import numpy as np
```
project starts here ;)
