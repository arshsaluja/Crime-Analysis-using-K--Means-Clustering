import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import folium
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

pip install folium

df=pd.read_csv('Downloads/20_Victims_of_rape.csv')

df

df.isna().sum()

df.describe

sns.pairplot(df[['Year','Rape_Cases_Reported','Victims_Above_50_Yrs','Victims_Between_14-18_Yrs','Victims_of_Rape_Total','Victims_Upto_10_Yrs']])

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
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Upto_10_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Upto_10_Yrs", y="Area_Name", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_10-14_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_10-14_Yrs", y="Area_Name", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_14-18_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_14-18_Yrs", y="Area_Name", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_18-30_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_18-30_Yrs", y="Area_Name", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_30-50_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Between_30-50_Yrs", y="Area_Name", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Above_50_Yrs", hue="Clusters",data=df)

plt.figure(figsize=(14,8))
sns.set(style="whitegrid", palette="muted")
sns.scatterplot(x="Victims_Above_50_Yrs", y="Area_Name", hue="Clusters",data=df)


data = pd.read_csv('Downloads\data.csv')
data

data.Magnitude.unique()

plt.scatter(data.Latitude,data.Longitude)
plt.xlabel('Latitude')
plt.ylabel('Longitude')

data.isna().sum()

arr = data.Magnitude.unique()
arr

arr = np.delete(arr, 7)
arr = arr.astype(int)

avg = np.average(arr)
avg

data.Magnitude = data.Magnitude.replace(to_replace ="ARSON", 
                 value ="9")

data['Magnitude'] = data['Magnitude'].astype(int)

X = data[['Latitude', 'Longitude', 'Magnitude']]
X = np.array(X)

X

wss = []
K = []
k_rng = range(1,50)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X)
    wss.append(km.inertia_)
    K.append(k)
plt.figure(figsize=(13,7)) 
plt.xlabel('K Values')
plt.ylabel('WSS')
axes= plt.axes()
axes.set_xticks(K)
plt.grid()
plt.plot(k_rng,wss)

k = KMeans(n_clusters=5, max_iter=150)
k.fit(X)
label = k.predict(X)

print(label)

filtered_label0 = X[label == 0]
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
plt.show()

filtered_label3 = X[label == 0]
 
filtered_label1 = X[label == 1]
 
#Plotting the results
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'red')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'yellow')
plt.show()

u_labels = np.unique(label)
 
#plotting the results:
plt.figure(figsize=(13,7)) 
for i in u_labels:
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
plt.legend()
plt.show()

latitude = 40.43457343
longitude = -79.32523454
map = folium.Map(location=[latitude, longitude], zoom_start=5)

map

for i in u_labels:
    for lat,lon in zip(X[label == i , 0] , X[label == i , 1]):
        folium.CircleMarker([lat,lon], radius = 1.0).add_to(map)
        
map

