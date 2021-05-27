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
# concatenate data 
files = os.listdir('world-happiness')
df = pd.DataFrame()
for file in files:
    df_file = pd.read_csv(os.path.join('world-happiness',file))
    df = pd.concat([df,df_file])

# %%
