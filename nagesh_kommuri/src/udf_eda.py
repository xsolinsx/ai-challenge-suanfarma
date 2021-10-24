import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_bind(fileNames):
    foo = []
    foo_BHV_CFF= []
    foo_EXT= []
    foo_NF = []
   
    col_names_list = []
    error_batches = []
    for i in tqdm(fileNames):
        data_BHV = pd.read_excel(i, sheet_name="BHV")
        data_BHV.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
        data_BHV['timeseries'] = pd.to_datetime(data_BHV['timeseries'], format = '%m-%d-%Y %H:%M:%S')
        data_BHV.insert(0,'id', i[49:53])
        data_BHV['id'] = data_BHV['id'].convert_dtypes()
        data_BHV = data_BHV[data_BHV.columns.drop(list(data_BHV.filter(regex='Unnamed:')))]
        
        data_CFF = pd.read_excel(i, sheet_name="CFF")
        data_CFF.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
        data_CFF['timeseries'] = pd.to_datetime(data_CFF['timeseries'], format = '%m-%d-%Y %H:%M:%S')
        data_CFF.insert(0,'id', i[49:53])
        data_CFF['id'] = data_CFF['id'].convert_dtypes()
        data_CFF = data_CFF[data_CFF.columns.drop(list(data_CFF.filter(regex='Unnamed:')))]
        
        data_EXT = pd.read_excel(i, sheet_name="EXT")
        data_EXT.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
        data_EXT['timeseries'] = pd.to_datetime(data_EXT['timeseries'], format = '%m-%d-%Y %H:%M:%S')
        data_EXT.insert(0,'id', i[49:53])
        data_EXT['id'] = data_EXT['id'].convert_dtypes()
        data_EXT = data_EXT[data_EXT.columns.drop(list(data_EXT.filter(regex='Unnamed:')))]
     
        data_NF = pd.read_excel(i, sheet_name="NF")
        data_NF.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
        data_NF['timeseries'] = pd.to_datetime(data_NF['timeseries'], format = '%m-%d-%Y %H:%M:%S')
        data_NF.insert(0,'id', i[49:53])
        data_NF['id'] = data_NF['id'].convert_dtypes()
        data_NF = data_NF[data_NF.columns.drop(list(data_NF.filter(regex='Unnamed:')))]
        
        try:
            data_BHV_CFF = data_BHV.merge(data_CFF)
            data = data_BHV_CFF.merge(data_EXT)
            data = data.merge(data_NF)
            #data.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
            #data['timeseries'] = pd.to_datetime(data['timeseries'], format = '%m-%d-%Y %H:%M:%S')
            #data.insert(0,'id', i[49:53])
            #data['id'] = data['id'].convert_dtypes()
            foo.append(data)
            foo_BHV_CFF.append(data_BHV_CFF)
            foo_EXT.append(data_EXT)
            foo_NF.append(data_NF)
            col_names_list.append(data.columns.values)
        except ValueError:
            error_batches.append(i[49:53])
            continue
        
    appended_data = pd.concat(foo)
    data_BHV_CFF = pd.concat(foo_BHV_CFF)
    data_EXT = pd.concat(foo_EXT)
    data_NF = pd.concat(foo_NF)
    
    
    
    
    col_names_list = [l.tolist() for l in col_names_list]
    
    
    # A simple data check
    print("The following batches have incompatible data: ", error_batches)
    print("# of batches read: ", appended_data.id.nunique())
    temp = list(map(lambda x: x[49:53], fileNames))
    print("Missing batches, if any:", set(temp).difference(set(list(appended_data.id.unique()) + error_batches)))

    # if any blank columns are created by accident in a spreadsheet software, which wouldn't have any column name, we remove such columns
    
    print("How many NaN values exist in the merged data: ", appended_data.isna().sum().sum())
    print("Shape of the merged data: ",appended_data.shape)

    return data_BHV_CFF,data_NF,data_EXT,appended_data

def describe(df):
    desc_df = df.describe(include = 'all', datetime_is_numeric=True)
    desc_df.loc['dtype'] = df.dtypes
    desc_df.loc['size'] = len(df)
    desc_df.loc['perc_null'] = df.isnull().mean()*100
    return desc_df

def drop_duplicate_columns(df):
    duplicate_cols = []
    for i in tqdm(df.columns):
        for j in df.columns:
            if ((str(i) != str(j)) & (df[i].equals(df[j]))):
                if (i not in duplicate_cols):
                    duplicate_cols.append(j)
    return df.drop(columns = duplicate_cols)

def plot_corr(df,size=10):
    # size: vertical and horizontal size of the plot
    corr = df.corr()
    corr = corr.dropna(how = 'all')
    corr = corr.dropna(how = 'all', axis=1)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 45)
    plt.yticks(range(len(corr.columns)), corr.columns)

def ecdf(data):
    """ Compute Empirical CDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

def eliminate_corr(cor_df, thresh):
    print("# of columns before dropping correlated variables: ", cor_df.shape[1])
    cor_matrix = cor_df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= thresh)]
    print("# of columns to drop", len(to_drop))
    print('Dropped columns after correlation analysis:',to_drop)
    cor_df = cor_df[cor_df.columns.drop(to_drop)]
    return cor_df