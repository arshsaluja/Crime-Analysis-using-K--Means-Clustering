import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.cluster as cluster
df=pd.read_csv('Downloads/20_Victims_of_rape.csv')
df
df.isna().sum()
df.describe
sns.pairplot(df[[Year','Rape_Cases_Reported','Victims_Above_50_Yrs','Victims_Between_14-18_Yrs','Victims_of_Rape_Total','Victims_Upto_10_Yrs']])
kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
kmeans=kmeans.fit(df[['Rape_Cases_Reported','Year']])
df['Clusters']=kmeans.labels_
df.head()
df['Clusters'].value_counts()
sns.scatterplot(x="Rape_Cases_Reported", y="Area_Name", hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Year", hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Upto_10_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Upto_10_Yrs", y="Area_Name",hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_10-14_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Between_10-14_Yrs", y="Area_Name",hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_14-18_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Between_14-18_Yrs", y="Area_Name",hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_18-30_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Between_18-30_Yrs", y="Area_Name",hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Between_30-50_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Between_30-50_Yrs", y="Area_Name",hue='Clusters', data=df)
sns.scatterplot(x="Rape_Cases_Reported", y="Victims_Above_50_Yrs", hue='Clusters', data=df)
sns.scatterplot(x="Victims_Above_50_Yrs", y="Area_Name",hue='Clusters', data=df)
                 
