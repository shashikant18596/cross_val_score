import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
lr = load_iris()
df = pd.DataFrame(lr.data,columns=lr.feature_names)
print(df.head())
df['target']=lr.target
print(df.head())
df['flowers_name'] = df.target.apply(lambda x: lr.target_names[x])
print(df.head())
model = LogisticRegression()
cvs = cross_val_score(model,df.drop(['target','flowers_name'],axis = 'columns'),df.target,cv=10)
print(cvs)
print(np.average(cvs))

