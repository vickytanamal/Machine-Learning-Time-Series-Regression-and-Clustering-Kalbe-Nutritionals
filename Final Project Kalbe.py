# %% [markdown]
# # <center> Final Project VIX Kalbe Nutritionals
# <center> by Vicky Tanamal

# %% [markdown]
# # Import Library

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Load Dataset

# %%
df_transaction = pd.read_csv('Case Study - Transaction.csv', sep=';')
df_store = pd.read_csv('Case Study - Store.csv', sep=';')
df_product = pd.read_csv('Case Study - Product.csv', sep=';')
df_customer = pd.read_csv('Case Study - Customer.csv', sep=';')

# %% [markdown]
# # Data Cleaning

# %%
# Overview data
print(df_transaction.head(3))
print(df_store.head(3))
print(df_product.head(3))
print(df_customer.head(3))

# %%
# Info data
print(df_transaction.info())
print(df_store.info())
print(df_product.info())
print(df_customer.info())

# %%
df_customer.isnull().sum()

# %% [markdown]
# There are missing values in Marital Status in df_customer. Because the missing values are in categorical column, we can use MODE to fill the NULL data.

# %%
df_customer = df_customer.apply(lambda x: x.fillna(x.mode()[0]))
df_customer.isnull().sum()

# %%
# Change data type
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'])
df_store['Latitude'] = df_store['Latitude'].str.replace(',','.').astype(float)
df_store['Longitude'] = df_store['Longitude'].str.replace(',','.').astype(float)
df_customer['Income'] = df_customer['Income'].str.replace(',','.').astype(float)

# %%
# Merge data
merged_df = pd.merge(df_transaction, df_store, on='StoreID', how='inner')
merged_df = pd.merge(merged_df, df_product, on='ProductID', how='inner')
merged_df = pd.merge(merged_df, df_customer, on='CustomerID', how='inner')

# %%
merged_df

# %%
# Drop same column
merged_df.drop('Price_y', axis=1, inplace=True)

# %%
merged_df

# %%
# Check Merged data
merged_df.info()

# %% [markdown]
# # Time Series Modelling

# %% [markdown]
# ## ARIMA Time Series

# %%
df_ts = merged_df.groupby('Date')['Qty'].sum().reset_index()
df_ts.set_index('Date',inplace=True)
df_ts

# %%
plt.figure(figsize=(12,6))
sns.lineplot(data=df_ts)
plt.title('Sales Qty in a Year')

# %%
# Splitting data train and test
print(df_ts.shape)
ts_train = df_ts.iloc[:-92] # First 9 months for training
ts_test = df_ts.iloc[-92:] # Last 3 months for testing
print(ts_train.shape,ts_test.shape)

# %%
ts_train.head()

# %%
ts_test.head()

# %%
# Testing For Stationarity
from statsmodels.tsa.stattools import adfuller

# %%
#Ho: It is non stationary
#H1: It is stationary

alpha = 0.05
adfuller_pvalue = adfuller(ts_train['Qty'])[1]

if adfuller_pvalue <= alpha:
    print('Reject Ho. The data is stationary')
    print(adfuller_pvalue)
else:
    print('Fail to reject Ho. The data is not stationary')
    print(adfuller_pvalue)

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_train,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_test,lags=40,ax=ax2)

# %%
model2=sm.tsa.statespace.SARIMAX(ts_train,order=(3, 0, 2),
                                 seasonal_order=(1,1,0,7))
results=model2.fit()
results.summary()

# %%
start = len(ts_train)
end = len(ts_train)+len(ts_test)-1
y_pred = results.predict(start=start, end=end, type='levels')
y_pred.index = ts_test.index
y_pred.head()

# %%
plt.figure(figsize=(12,6))
plt.plot(y_pred, label='Predictions')
plt.plot(ts_test, label='Test Data')
plt.title('Actual Data vs Predicions')
plt.legend()

# %%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def rmse(y_actual, y_pred):
  print(f'RMSE Value: {mean_squared_error(y_actual, y_pred)**0.5}')
def rsquare(y_actual, y_pred):
  print(f'R-squared Value: {r2_score(y_actual, y_pred)}')
def eval(y_actual, y_pred):
  rmse(y_actual, y_pred)
  rsquare(y_actual, y_pred)
  print(f'MAE Value: {mean_absolute_error(y_actual, y_pred)}')

# %%
eval(ts_test['Qty'], y_pred)

# %% [markdown]
# ### Forecast

# %%
index_future_dates = pd.date_range(start='2022-12-31', end='2023-12-31')
final_pred = results.predict(start=len(df_ts), end=len(df_ts)+365, type='levels').rename('ARIMA Predictions')
final_pred.index = index_future_dates

# %%
final_pred.plot(figsize=(16,6), legend=True)

# %%
future_df.tail()

# %%
future_df.rename(columns={0:'Forecast'}, inplace=True)

# %%
future_df.plot(figsize=(16, 8), legend=True, title='SARIMA Predictions')

# %% [markdown]
# # Clustering Modelling

# %%
clustering_df = merged_df.groupby('CustomerID').agg({'TransactionID':'count',
                                                     'Qty':'sum',
                                                     'TotalAmount':'sum'}).reset_index()
clustering_df

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
# Standarisasi Data
features = clustering_df.iloc[:, 1:]
fs_cols = features.columns

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_features_scaled = pd.DataFrame(data=features_scaled, columns=fs_cols)
df_features_scaled

# %%
# Using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# %% [markdown]
# ### Clustering with n=4

# %%
clusters = 4

# Fit K-Means model
kmeans = KMeans(n_clusters=clusters, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)

# Get cluster assignments for each data point
clustering_df['Cluster'] = cluster_labels

# %%
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(clustering_df, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# %% [markdown]
# ### Clustering with n=3

# %%
clusters = 3

# Fit K-Means model
kmeans = KMeans(n_clusters=clusters, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)

# Get cluster assignments for each data point
clustering_df['Cluster'] = cluster_labels

# %%
silhouette_avg = silhouette_score(clustering_df, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# %% [markdown]
# Silhouette score when n = 3 is higher than n = 4, so we use n = 3 for clustering.

# %%
sns.scatterplot(data=clustering_df, x='Qty', y='TotalAmount', hue='Cluster', palette='Set1')
plt.title('K-Means Clusters')
plt.show()

# %%
clustering_df.head()

# %%
# Counting customer per cluster
cl_count = clustering_df['Cluster'].value_counts().reset_index()
cl_count.columns = ['Cluster','Count']
cl_count['Percentage(%)'] = round((cl_count['Count']/len(clustering_df))*100, 2)
cl_count = cl_count.sort_values(by=['Cluster']).reset_index(drop=True)
cl_count

# %%
sns.barplot(data=cl_count, x='Cluster', y='Percentage(%)', palette='Set1')
plt.title('Percentage of Customer by Cluster')

# %% [markdown]
# # Customer Segmentation Analysis

# %%
clustering_df = clustering_df.drop('CustomerID', axis=1)
clustering_df.groupby('Cluster').agg(['min', 'max', 'mean']).reset_index(drop = True).T

# %% [markdown]
# Although Cluster 1 have the most customer, but Cluster 0 has higher mean for Qty, Total Amount and Transaction. Meanwhile Cluster 2 has the lowest mean for Qty, Total Amount and Transaction.

# %% [markdown]
# # Business Recommendation

# %% [markdown]
# 1. Cluster 0
# <br>We must keep this customer, because this Cluster has high value. We can give them like Loyalty Programs for repeat purchases or buy product by passing the limit of shopping and on that program we can give some points and the points can be exchanged with our another product for free.

# %% [markdown]
# 2. Cluster 1
# <br>Most of customer in this Cluster, so we must increase buying rate of the customer. We can give them discount voucher after they bought product, so they consider to buy another product using that voucher.

# %% [markdown]
# 3. Cluster 2
# <br>We must do some campaigns that can make our products become their top of mind to increase the buying rate. We must give them knowledge our product, why must choose and buy our product and we can highlight the good review for our product to proof that our product is good and worth to buy. 


