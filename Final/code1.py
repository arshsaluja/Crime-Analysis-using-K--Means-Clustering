import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import folium
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
df=pd.read_csv('Downloads/20_Victims_of_rape.csv')
df
df.isna().sum()
df.describe
sns.pairplot(df[['Year','Rape_Cases_Reported','Victims_Above_50_Yrs','Victims_Between_1
4-18_Yrs','Victims_of_Rape_Total','Victims_Upto_10_Yrs']])
mms = StandardScaler()
mms.fit(df[['Rape_Cases_Reported','Year']])
normalized_data = mms.transform(df[['Rape_Cases_Reported','Year']])
K = range(1,10)
Sum_of_squared_distances = []
for k in K:
 km = KMeans(n_clusters=k)
 km = km.fit(normalized_data)
 Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('No of Clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
kmeans=kmeans.fit(df[['Rape_Cases_Reported','Year']])
kmeans.cluster_centers_
df['Clusters']=kmeans.labels_
df.head()
df['Clusters'].value_counts()
15
fig = plt.figure(figsize=(14,10))
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.boxplot(x="Year", y="Rape_Cases_Reported", data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Area_Name", hue="Clusters",data=df)
fig = plt.figure(figsize=(14,10))
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.boxplot(x="Rape_Cases_Reported",y="Area_Name", data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Year", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Upto_10_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Upto_10_Yrs", y="Area_Name", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_10-14_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_10-14_Yrs", y="Area_Name", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_14-18_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_14-18_Yrs", y="Area_Name", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_18-30_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
16
sns.scatterplot(x="Victims_Between_18-30_Yrs", y="Area_Name", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_30-50_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_30-50_Yrs", y="Area_Name", hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Above_50_Yrs", 
hue="Clusters",data=df)
plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Above_50_Yrs", y="Area_Name", hue="Clusters",data=df)
