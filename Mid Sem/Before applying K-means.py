import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.cluster as cluster
df=pd.read_csv('../Downloads/20_Victims_of_rape.csv')
df
df.isna().sum()
df.describe
sns.pairplot(df[['Year','Rape_Cases_Reported','Victims_Above_50_Yrs','Victims_Between_14-18_Yrs','Victims_of_Rape_Total','Victims_Upto_10_Yrs']])
