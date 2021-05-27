#%% import various libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
import os
# %% 
# import data from kaggle 
api = KaggleApi()
api.authenticate()
api.dataset_download_files('unsdsn/world-happiness')

#%%
# unzip data 
zf = ZipFile('world-happiness.zip')
os.makedirs('world-happiness')
zf.extractall('world-happiness/') # save files in selected folder
zf.close()
# %%
# becuase the data columns have different names in each year, we will only use the income column and the life expectency column as an example
files = os.listdir('world-happiness')
df = pd.DataFrame()
for file in files:
    df_file = pd.read_csv(os.path.join('world-happiness',file))
    ## get the columns we need - country, gdp and life expectancy
    cols_get = [x for x in df_file.columns.tolist() if 'gdp' in x.lower() or 'expectancy' in x.lower()or 'country' in x.lower()]
    ## sort and rename the columns for consistancy 
    cols_get.sort()
    dict_rename = dict(zip(cols_get,['Country','GDP','Life Expectancy']))
    df_file = df_file[cols_get].copy()
    df_file.rename(columns=dict_rename,inplace=True)
    ## add the year column for reference 
    df_file['year'] = os.path.splitext(file)[0]
    ## concatenate the data 
    df = pd.concat([df,df_file])
    

# %%
# the problem is if we can find a linear relation between GDP and life expectancy 
# we will first look into some basic stats of the data to determine if we need process the data before feeding it to a model 
df.describe()
# want to make sure the distribution in each year is relatively the same so they can be trained as a whole dataset 
# aka: we want to make sure time does not affect the data very much 
df.boxplot(column=['GDP','Life Expectancy'],by=['year'],figsize=(12,5))

# here we can see that the two variables have different maximum values, yet their distributions are quite similar judging from the percentiles. We will just scale the data to make the range of the variables 0-1 in this case 
# in addition, based on the boxplot grouped by year we can see that time is not an obvious influence factor here. So we can use the data as a whole 
#%%
## import normalizer from sklearn 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform data
X = df['GDP'].values
X_scaled = scaler.fit_transform(X.reshape(-1,1))
y = df['Life Expectancy'].values
y_scaled = scaler.fit_transform(y.reshape(-1,1))

# %%
random_state = 17
# start building the model 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=random_state)
# %%
# build the model 
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
#%%
from sklearn.metrics import mean_squared_error
print('Model RMSE on training data set: {:.2f}'.format(mean_squared_error(reg.predict(X_train),y_train)))
print('Model RMSE on test data set: {:.2f}'.format(mean_squared_error(reg.predict(X_test),y_test)))
# %%
## evaluate the model by visualization 
X_model = np.linspace(0,1,100)
y_model = reg.predict(X_model.reshape(-1,1))
plt.figure(figsize=(12,5))
plt.scatter(X_train,y_train,color='tab:blue',label='train')
plt.scatter(X_test,y_test,color='tab:orange',label='test')
plt.plot(X_model,y_model, 'r--',label='fitted')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('GDP vs Life Expectancy')
plt.legend()

# from the figure we can see that there is a clear trend as the higher the GDP is, the larger the life expectancy is. However, correlation does not necessarily mean causation. We will not dive deep into this topic here. 
# %%
## some statistical exploration 
## TBC 
## prediction interval, confidence interval 