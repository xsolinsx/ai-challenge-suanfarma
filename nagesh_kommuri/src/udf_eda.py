import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_bind(fileNames):
    foo = []
    col_names_list = []
    error_batches = []
    for i in tqdm(fileNames):
        data0 = pd.read_excel(i, sheet_name="BHV")
        data1 = pd.read_excel(i, sheet_name="CFF")
        data2 = pd.read_excel(i, sheet_name="EXT")
        data3 = pd.read_excel(i, sheet_name="NF")
        
        try:
            data = data0.merge(data1)
            data = data.merge(data2)
            data = data.merge(data3)
            data.rename(columns = {'Unnamed: 0' : 'timeseries'}, inplace = True)
            data['timeseries'] = pd.to_datetime(data['timeseries'], format = '%m-%d-%Y %H:%M:%S')
            data.insert(0,'id', i[49:53])
            data['id'] = data['id'].convert_dtypes()
            foo.append(data)
            col_names_list.append(data.columns.values)
        except ValueError:
            error_batches.append(i[49:53])
            continue
        
    appended_data = pd.concat(foo)
    col_names_list = [l.tolist() for l in col_names_list]
    # A simple data check
    print("The following batches have incompatible data: ", error_batches)
    print("# of batches read: ", appended_data.id.nunique())
    temp = list(map(lambda x: x[49:53], fileNames))
    print("Missing batches, if any:", set(temp).difference(set(list(appended_data.id.unique()) + error_batches)))

    return appended_data

def describe(df):
    desc_df = df.describe(include = 'all')
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