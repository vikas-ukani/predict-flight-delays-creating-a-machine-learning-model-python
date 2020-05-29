# predict-flight-delays-creating-a-machine-learning-model-python

#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')


# In[ ]:





# In[2]:


import pandas as pd


# In[4]:


df = pd.read_csv('flightdata.csv')
df.head()


# In[8]:


df.shape


# In[13]:


# Check for empty or null values availables
df.isnull().values.any()


# In[14]:


df.isnull().sum()


# In[15]:


#  Drop
df = df.drop('Unnamed: 25', axis=1)
df.isnull().sum()


# In[16]:


#  Filter columns


# In[17]:


df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
df.isnull().sum()


# In[20]:


#  Full 0 to ARR_DEL15 column
df[df.isnull().values.any(axis=1)].head()


# In[23]:


#  Fill 1 to all null values
df = df.fillna({"ARR_DEL15": 1})
df.iloc[177:185]


# In[25]:


df.head()


# In[26]:


#  Binning the depature time
import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row[ 'CRS_DEP_TIME'] / 100 )
df.head()


# In[73]:


df = pd.get_dummies(df, columns=['ORIGIN' , 'DEST'])
df.head() 


# In[ ]:





# 
# # Build Machine Learning Model

# In[74]:


# Load package
from sklearn.model_selection import train_test_split



# In[75]:


# Train model
X = df.drop('ARR_DEL15', axis=1)
y = df['ARR_DEL15']
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)


# In[76]:


print(train_x.shape)
print(train_y.shape)


# In[77]:


print(test_x.shape)
print(test_y.shape)


# # RandomForestClassifire Model Applying
# 

# In[78]:


from sklearn.ensemble import RandomForestClassifier
# import warnings

# warnings.filterwarnings('ignore')


# In[79]:


#  Fit the data
model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)


# In[80]:


#  Predict Model
predicted = model.predict(test_x)
model.score(test_x, test_y)


# In[ ]:





# #  Area Under Receiver Operating Characteristic Curve Applying

# In[81]:


from sklearn.metrics import roc_auc_score


# In[82]:


probabilities = model.predict_proba(test_x)


# In[84]:


# Generating an AUC score


roc_auc_score(test_y, probabilities[:, 1])


# In[ ]:





# #  produce a confusion matrix

# In[85]:


from sklearn.metrics import confusion_matrix


# In[86]:


confusion_matrix(test_y, predicted)


# In[88]:


# Measuring precision


from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
precision_score(train_y, train_predictions)


# In[89]:


# Masering Recall score
from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)


# In[ ]:





# # Visualize Output of Model

# In[95]:


# Import packages for visulaize data

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import seaborn as sns
sns.set()


# In[100]:


#  Create first plot

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])


# ROC curve generated with Matplotlib
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], color='grey', lw=1, linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# In[102]:



def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]


# In[107]:


predict_delay('01/05/2020 21:45:00', 'JFK', 'ATL')


# In[108]:


predict_delay('2/10/2018 10:00:00', 'ATL', 'SEA')


# In[118]:


# Use differ
import numpy as np

label = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(label))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, label)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))
plt.title('Probability of on-time arrivals for a range of dates')
plt.show()


# In[ ]:




