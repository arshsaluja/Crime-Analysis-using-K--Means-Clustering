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
18
map
for i in u_labels:
 for lat,lon in zip(X[label == i , 0] , X[label == i , 1]):
 folium.CircleMarker([lat,lon], radius = 1.0).add_to(map)
 
map
