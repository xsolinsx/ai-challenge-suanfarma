import pandas as pd
from tqdm import tqdm


def read_bind(fileNames):
    foo = list()
    foo_BHV_CFF = list()
    foo_EXT = list()
    foo_NF = list()

    col_names_list = list()
    error_batches = list()
    for i in tqdm(fileNames):
        # take last part of path, take first part of filename, remove "ODP ", remove .xlsx if present and strip spaces
        batch_name = i.split("/")[-1].split("_")[0][4:].replace(".xlsx", "").strip()
        data_BHV = pd.read_excel(i, sheet_name="BHV")
        data_BHV.rename(columns={"Unnamed: 0": "timeseries"}, inplace=True)
        data_BHV["timeseries"] = pd.to_datetime(
            data_BHV["timeseries"], infer_datetime_format=True
        )
        data_BHV.insert(0, "id", batch_name)
        data_BHV["id"] = data_BHV["id"].convert_dtypes()
        data_BHV = data_BHV[
            data_BHV.columns.drop(list(data_BHV.filter(regex="Unnamed:")))
        ]

        data_CFF = pd.read_excel(i, sheet_name="CFF")
        data_CFF.rename(columns={"Unnamed: 0": "timeseries"}, inplace=True)
        data_CFF["timeseries"] = pd.to_datetime(
            data_CFF["timeseries"], infer_datetime_format=True
        )
        data_CFF.insert(0, "id", batch_name)
        data_CFF["id"] = data_CFF["id"].convert_dtypes()
        data_CFF = data_CFF[
            data_CFF.columns.drop(list(data_CFF.filter(regex="Unnamed:")))
        ]

        data_EXT = pd.read_excel(i, sheet_name="EXT")
        data_EXT.rename(columns={"Unnamed: 0": "timeseries"}, inplace=True)
        data_EXT["timeseries"] = pd.to_datetime(
            data_EXT["timeseries"], infer_datetime_format=True
        )
        data_EXT.insert(0, "id", batch_name)
        data_EXT["id"] = data_EXT["id"].convert_dtypes()
        data_EXT = data_EXT[
            data_EXT.columns.drop(list(data_EXT.filter(regex="Unnamed:")))
        ]

        data_NF = pd.read_excel(i, sheet_name="NF")
        data_NF.rename(columns={"Unnamed: 0": "timeseries"}, inplace=True)
        data_NF["timeseries"] = pd.to_datetime(
            data_NF["timeseries"], infer_datetime_format=True
        )
        data_NF.insert(0, "id", batch_name)
        data_NF["id"] = data_NF["id"].convert_dtypes()
        data_NF = data_NF[data_NF.columns.drop(list(data_NF.filter(regex="Unnamed:")))]

        try:
            data_BHV_CFF = data_BHV.merge(data_CFF)
            data = data_BHV_CFF.merge(data_EXT)
            data = data.merge(data_NF)
            foo.append(data)
            foo_BHV_CFF.append(data_BHV_CFF)
            foo_EXT.append(data_EXT)
            foo_NF.append(data_NF)
            col_names_list.append(data.columns.values)
        except ValueError:
            error_batches.append(batch_name)
            continue

    appended_data = pd.concat(foo)
    data_BHV_CFF = pd.concat(foo_BHV_CFF)
    data_EXT = pd.concat(foo_EXT)
    data_NF = pd.concat(foo_NF)

    col_names_list = [li.tolist() for li in col_names_list]

    # A simple data check
    print("The following batches have incompatible data: ", error_batches)
    print("# of batches read: ", appended_data.id.nunique())
    temp = list(map(lambda x: x[49:53], fileNames))
    print(
        "Missing batches, if any:",
        set(temp).difference(set(list(appended_data.id.unique()) + error_batches)),
    )

    # if any blank columns are created by accident in a spreadsheet software, which wouldn't have any column name, we remove such columns
    print(
        "How many NaN values exist in the merged data: ",
        appended_data.isna().sum().sum(),
    )
    print("Shape of the merged data: ", appended_data.shape)

    return data_BHV_CFF, data_NF, data_EXT, appended_data
