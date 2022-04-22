# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:24:32 2022

@author: ankush
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:46:28 2022

@author: ankush
"""

import pandas as pd

# we use autoinsurance dataset
df = pd.read_csv("C:/Users/ankush/Desktop/DataSets/H-Clustr/AutoInsurance.csv")

df.columns # column names
df.shape # will give u shape of the dataframe

#take categorical data into one file for lable encoding
insu=df[['Coverage','EmploymentStatus','Location Code','Policy Type','Policy','Renew Offer Type']]
insu.columns

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = insu.iloc[:, 0:6]

y = insu['Renew Offer Type']
y = insu.iloc[:, 6:] # Alternative approach

insu.columns

X['Coverage'] = labelencoder.fit_transform(X['Coverage'])
X['EmploymentStatus'] = labelencoder.fit_transform(X['EmploymentStatus'])
X['Location Code'] = labelencoder.fit_transform(X['Location Code'])
X['Policy Type'] = labelencoder.fit_transform(X['Policy Type'])
X['Policy'] = labelencoder.fit_transform(X['Policy'])
X['Renew Offer Type'] = labelencoder.fit_transform(X['Renew Offer Type'])

### label encode y ###
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)
insu_new = pd.concat([X, y], axis =1)

insu_new.columns
insu_new = insu_new.rename(columns={0:'Renew Offer Type'})


#take numerical data for normalization
insu1=df[['Customer','Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Policies','Total Claim Amount']]

insu1.describe()
insu1.info()

insu1 = insu1.drop(["Customer"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insu1.iloc[:, 1:])
df_norm.describe()

insu_new1 = pd.concat([insu_new , df_norm], axis =1)
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt
z = linkage(insu_new1, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

insu_new1['clust'] = cluster_labels # creating a new column and assigning it to new column 

insu_new1.head()

# Aggregate mean of each cluster
df_mean=df.iloc[:,].groupby(insu_new1.clust).mean()

# shift column 'Name' to first position
first_column =insu_new1.pop('clust')
  
# insert column using insert(position,column_name,
# first_column) function
df.insert(0, 'clust', first_column)
# creating a csv file 
df.to_csv("insurance1.csv", encoding = "utf-8")

import os
os.getcwd()
