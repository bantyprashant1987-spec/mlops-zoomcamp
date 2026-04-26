#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:





# In[ ]:


import pickle
import pandas as pd
import sys


# In[ ]:
year = int(sys.argv[1])
month = int(sys.argv[2])

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[ ]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[ ]:


#df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_????-??.parquet')

df = read_data(input_file)
# In[ ]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:




print(f'The mean predicted duration for {year}/{month} is: {y_pred.mean():.2f}')