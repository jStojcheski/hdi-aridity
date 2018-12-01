def file_to_df(path, skiprows=0):
    df = pd.read_csv(path, index_col='Country', skiprows=skiprows)
    if 'HDI Rank' in df.columns:
        df = df.drop('HDI Rank', axis=1)
    if 'HDI Rank (2015)' in df.columns:
        df = df.drop('HDI Rank (2015)', axis=1)
    return df