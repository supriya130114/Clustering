# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 12:37:11 2022

@author: ankush
"""


import pandas as pd

# we use autoinsurance dataset
telecom = pd.read_excel("C:/Users/ankush/Desktop/DataSets/H-Clustr/Telco_customer_churn.xlsx")

telecom.columns # column names
telecom.shape # will give u shape of the dataframe

#take categorical data into one file for lable encoding
telecom1=telecom.drop(['Count', 'Quarter', 'Referred a Friend','Number of Referrals', 'Tenure in Months','Contract', 'Paperless Billing', 'Payment Method'],axis=1)
telecom1.columns

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
telecomm2_norm = norm_func(telecom1.iloc[:, 16:])
                           

telecomm3_norm = norm_func(telecom1.iloc[:, telecom1.columns.get_loc('Avg Monthly Long Distance Charges')])
telecomm4_norm = norm_func(telecom1.iloc[:, telecom1.columns.get_loc('Avg Monthly GB Download')])

telecomm_new = pd.concat([telecomm2_norm, telecomm3_norm,telecomm4_norm], axis =1)
telecomm_new.columns
telecomm_new.shape
telec=telecom1[['Offer', 'Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data']]
telec.dtypes
tel=pd.concat([telec, telecomm_new ], axis =1)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(telecomm_new, method = "complete", metric = "euclidean")
import matplotlib.pylab as plt
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(telecomm_new) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

telecomm_new['clust'] = cluster_labels # creating a new column and assigning it to new column 
telecomm_new.head
telecomm_new.shape

# Aggregate mean of each cluster
telemean=telecom1.iloc[:,].groupby(telecomm_new.clust).mean()
telemean.shape
telemean.head
 

# shift column 'Name' to first position
first_column =telecomm_new.pop('clust')
  
# insert column using insert(position,column_name,
# first_column) function
telecom.insert(0, 'clust', first_column)
# creating a csv file 
telecom.to_csv("telecommchurn.csv", encoding = "utf-8")
import os
os.getcwd()
